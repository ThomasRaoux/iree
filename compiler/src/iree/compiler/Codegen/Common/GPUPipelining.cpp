// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <iostream>

//====---------------------------------------------------------------------===//
// Pass to pipeline copy to shared memory for matmul op.
//====---------------------------------------------------------------------===//

namespace mlir {
namespace iree_compiler {

static const StringLiteral kPipeliningLoopMarker = "__pipelining_K_loop__";
static const StringLiteral kPipeliningGlobalLoad = "__pipelining_global_load__";
static const StringLiteral kPipeliningLdmatrix = "__pipelining_ldmatrix__";

/// Helper to recursively add operation dependencies within `block` to `dep`
/// set.
static void addDepOps(llvm::SmallDenseSet<Operation*>& dep, Operation* op,
                      Block* block) {
  if (!dep.insert(op).second) return;
  for (Value operand : op->getOperands()) {
    Operation* defOp = operand.getDefiningOp();
    if (defOp && defOp->getBlock() == block) addDepOps(dep, defOp, block);
  }
}

/// Assign stages to the loop ops. Simple logic for now, put load from global
/// memory in stage 0 and the rest in stage 1.
static void getPipelineStages(scf::ForOp forOp,
                              std::vector<std::pair<Operation*, unsigned>>& ops,
                              unsigned depth) {
  if (!forOp->hasAttr(kPipeliningLoopMarker)) return;

  // Track dependencies of the global memory load.
  llvm::SmallDenseSet<Operation*> loadDep;
  llvm::SmallDenseSet<Operation*> ldMatrixDep;
  for (Operation& op : forOp.getBody()->getOperations()) {
    if (op.hasAttr(kPipeliningGlobalLoad)) {
      addDepOps(loadDep, &op, forOp.getBody());
    }
    if (op.hasAttr(kPipeliningLdmatrix)) {
      addDepOps(ldMatrixDep, &op, forOp.getBody());
    }
  }
  
  // Create a modulo schedule with loads from global memory and the operations
  // it depends on in stage 0. Store to shared memory and computation are in
  // stage `maxDepth`. In order to have a correct scheduling even with back
  // edges we order stages in decreasing order.
  for (Operation& op : forOp.getBody()->getOperations()) {
    if (loadDep.count(&op)) ops.push_back(std::make_pair(&op, 0));
  } 
  for (Operation& op : forOp.getBody()->getOperations()) {
    if (!loadDep.count(&op) && !ldMatrixDep.count(&op) &&
        !isa<scf::YieldOp>(op))
      ops.push_back(std::make_pair(&op, depth));
  }
  for (Operation& op : forOp.getBody()->getOperations()) {
    if (ldMatrixDep.count(&op) && !loadDep.count(&op))
      ops.push_back(std::make_pair(&op, depth - 1));
  }

}

static void setAsyncAnnotations(Operation* op,
                                scf::PipeliningOption::PipelinerPart part,
                                unsigned iteration, unsigned depth) {
  auto waitOp = dyn_cast<nvgpu::DeviceAsyncWaitOp>(op);
  if (!waitOp || waitOp.getNumGroups()) return;
  int numGroupInFlight = 0;
  if (part == scf::PipeliningOption::PipelinerPart::Kernel) {
    numGroupInFlight = depth - 2;
  } else {
    // By construction there should be no wait op in the prologue as all the
    // wait should be in the last stage.
    if(part == scf::PipeliningOption::PipelinerPart::Prologue) {
        numGroupInFlight = depth - 2;
    } else {
    // Based on the schedule we pick we know how many groups are in flight for
    // each iteration of the epilogue.
    numGroupInFlight = depth - 3 - iteration;
    }
  }
  OpBuilder b(op);
  waitOp->setAttr(waitOp.getNumGroupsAttrName(),
                  b.getI32IntegerAttr(numGroupInFlight));
}

// Returns a new AsyncCopyOp with Zfill 
static Operation* replaceAsyncCopywithAsyncCopyZfill(Operation* op, Value pred, PatternRewriter& rewriter) {

  if (!isa<nvgpu::DeviceAsyncCopyOp>(op)) return op;

  // replace mainloop AsyncCopy with AsyncCopy(zfill) inline asm.
  auto asyncCopyOp = dyn_cast<nvgpu::DeviceAsyncCopyOp>(op);
  auto loc = asyncCopyOp->getLoc();

  // create srcElement Value based on the pred
  // srcElement = (pred) ?  dstElements : 0;

  Value dstElements = 
      rewriter.create<mlir::LLVM::ConstantOp>(loc, rewriter.getI32Type(), 
        asyncCopyOp.getDstElements());

  Value c0I32 =
      rewriter.create<mlir::LLVM::ConstantOp>(loc, rewriter.getI32Type(), 
        rewriter.getI32IntegerAttr(0));

  auto srcElements = rewriter.create<arith::SelectOp>(loc, pred, dstElements, c0I32);

  auto asyncCopyZfillOp = rewriter.create<nvgpu::DeviceAsyncCopyOp>(
          loc,
          nvgpu::DeviceAsyncTokenType::get(asyncCopyOp.getContext()),
          asyncCopyOp.getDst(), asyncCopyOp.getDstIndices(), 
          asyncCopyOp.getSrc(), asyncCopyOp.getSrcIndices(),
          asyncCopyOp.getDstElements(),
          srcElements,
          UnitAttr());
  
  rewriter.eraseOp(asyncCopyOp);

  // return the newly create AsyncCopyZfillOp
  return asyncCopyZfillOp;
}

namespace {
struct GPUPipeliningPass : public GPUPipeliningBase<GPUPipeliningPass> {
  GPUPipeliningPass(unsigned depth) : depth(depth) {}
  void runOnOperation() override {
    auto funcOp = getOperation();
    MLIRContext* context = &getContext();
    // Mark the loop with shared memory copy for pipelining.
    funcOp.walk([](scf::ForOp forOp) {
      bool copyToWorkgroupMemory = false;
      OpBuilder builder(forOp.getContext());
      SmallVector<Operation*> barriers;
      bool waitFound = false;
      int ldmatrixCounter = 0;           // OperandA
      int ldmatrixTransposeCounter = 0;  // OperandB
      for (Operation& op : forOp.getBody()->getOperations()) {
        // Pipeline the most inner for op that should be a flat region.
        if (op.getNumRegions() > 0) return;
        if (isa<gpu::BarrierOp>(op)) {
          if(waitFound) {
            op.setAttr(kPipeliningLdmatrix, builder.getUnitAttr());
          } else {
           barriers.push_back(&op);
          }
        }
        if (isa<nvgpu::DeviceAsyncCopyOp, nvgpu::DeviceAsyncCreateGroupOp>(
                op)) {
          copyToWorkgroupMemory = true;
          op.setAttr(kPipeliningGlobalLoad, builder.getUnitAttr());
          // async copy ops need to be moved along with previous barrier.
          for (Operation* barrier : barriers) {
            barrier->setAttr(kPipeliningGlobalLoad, builder.getUnitAttr());
          }
          barriers.clear();
          continue;
        }
        if (isa<nvgpu::DeviceAsyncWaitOp>(op)) {
          waitFound = true;
          op.setAttr(kPipeliningLdmatrix, builder.getUnitAttr());
        }
        if (isa<nvgpu::LdMatrixOp>(op)) {

          auto ldMatrixOp = cast<nvgpu::LdMatrixOp>(op);

          if (ldMatrixOp.getTranspose() && ldmatrixTransposeCounter < 4) {
            op.setAttr(kPipeliningLdmatrix, builder.getUnitAttr());
            ldmatrixTransposeCounter++;
          }
          
          else if (ldmatrixCounter < 4) {
            op.setAttr(kPipeliningLdmatrix, builder.getUnitAttr());
            ldmatrixCounter++;
          }
        }
        auto ld = dyn_cast<vector::TransferReadOp>(op);
        if (!ld) continue;
        unsigned ldAddSpace =
            ld.getSource().getType().cast<MemRefType>().getMemorySpaceAsInt();
        if (ldAddSpace != 0 || !ld->hasOneUse()) continue;
        auto st =
            dyn_cast<vector::TransferWriteOp>(ld->use_begin()->getOwner());
        if (!st) continue;
        unsigned stAddSpace =
            st.getSource().getType().cast<MemRefType>().getMemorySpaceAsInt();
        if (stAddSpace != 3) continue;
        copyToWorkgroupMemory = true;
        ld->setAttr(kPipeliningGlobalLoad, builder.getUnitAttr());
      }
      if (copyToWorkgroupMemory) {
        forOp->setAttr(kPipeliningLoopMarker, builder.getUnitAttr());
      }
    });
    scf::PipeliningOption options;

    unsigned maxDepth = depth; 
    auto getSchedule = [maxDepth](
                           scf::ForOp forOp,
                           std::vector<std::pair<Operation*, unsigned>>& ops) {
      return getPipelineStages(forOp, ops, maxDepth - 1);
    };
    auto setAnnotation = [maxDepth](Operation* op,
                                    scf::PipeliningOption::PipelinerPart part,
                                    unsigned iteration) {
      return setAsyncAnnotations(op, part, iteration, maxDepth);
    };
    options.getScheduleFn = getSchedule;
    options.annotateFn = setAnnotation;

    // Turn on/off epilogue peeling
    options.peelEpilogue = false;
    options.predicateFn = [](Operation* op, Value pred, PatternRewriter& rewriter) {
      return replaceAsyncCopywithAsyncCopyZfill(op, pred, rewriter);
    };

    RewritePatternSet pipeliningPatterns(context);
    scf::populateSCFLoopPipeliningPatterns(pipeliningPatterns, options);
    if (failed(applyPatternsAndFoldGreedily(funcOp,
                                            std::move(pipeliningPatterns)))) {
      return signalPassFailure();
    }
    
    // Rearrange ldmatrix closer to the mma.sync
    // std::cout << "GPUPipeline post processing " << std::endl;
#if 0
    funcOp.walk([](scf::ForOp forOp) {
      llvm::SmallDenseSet<Operation*, 32> ldMatrixOpSet;
      for (Operation& op : forOp.getBody()->getOperations()) {
        if (isa<nvgpu::LdMatrixOp>(op)) {
          ldMatrixOpSet.insert(&op);
        }
        else {

          for (auto operand : op.getOperands()) {
            Operation* defOp = operand.getDefiningOp();
            if (ldMatrixOpSet.contains(defOp)) {
              // if the defining operation is present in ldMatrixOpSet move 
              // it just above its use.
              defOp->moveBefore(&op);
              // only move each ldmatrix once, just before its first use.
              ldMatrixOpSet.erase(defOp);
            }
          }
        }
      }
      // std::cout << "unmoved nvgpu.ldmatrix instructions " << ldMatrixOpSet.size() << std::endl;
    });
#endif
  }

 private:
  unsigned depth;
};
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createGPUPipeliningPass(
    unsigned depth) {
  return std::make_unique<GPUPipeliningPass>(depth);
}

}  // namespace iree_compiler
}  // namespace mlir
