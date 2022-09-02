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

/// Helper to move operations before a specific Op to achive better schedule. 
static void moveOperationsBefore(
  Operation* movingOp, 
  std::vector<Operation*>& userOps, 
  Operation* beforeOp)
{
  
  // Move op 
  movingOp->moveBefore(beforeOp);

  // Move all the extract uses on ldsm op
  for (auto &op : userOps) {
    op->moveBefore(beforeOp);
  }
}

static void traverseDefChainUtilSharedMemoryLoad(
  Operation* op, llvm::SetVector<Operation*>& opSeq,
  Block* block) {
  
  if (!op) return;
  
  if (isa<nvgpu::LdMatrixOp>(op)) {
    if (op->getBlock() == block) opSeq.insert(op);
    return;
  }

  // Recurse upwards towards the definition until a load is found.
  // Assumption here is that only single operand operations are leading up to LdMatrix.
  Operation* defOp = op->getOperand(0).getDefiningOp();
    
  traverseDefChainUtilSharedMemoryLoad(defOp, opSeq, block);
}

/// Helper function to backtrack from MmaSyncOp and populate instructions uptil LdMatrixOp
/// feeding into Lhs and Rhs
static void getMmaSyncLoadSequence(
  nvgpu::MmaSyncOp mmaOp, 
  llvm::SetVector<Operation*>& lhsSeq, // Operations feeding into mmaOp lhs operandA registers
  llvm::SetVector<Operation*>& rhsSeq  // Operations feeding into mmaOp rhs operandB registers
  ) {

  Operation* lhs = mmaOp->getOperand(0).getDefiningOp();
  traverseDefChainUtilSharedMemoryLoad(lhs, lhsSeq, mmaOp->getBlock());
  
  Operation* rhs = mmaOp->getOperand(1).getDefiningOp();
  traverseDefChainUtilSharedMemoryLoad(rhs, rhsSeq, mmaOp->getBlock());
}

/// Assign stages to the loop ops. Simple logic for now, put load from global
/// memory in stage 0 and the rest in stage 1.
static void getPipelineStages(scf::ForOp forOp,
                              std::vector<std::pair<Operation*, unsigned>>& ops,
                              unsigned depth) {
  if (!forOp->hasAttr(kPipeliningLoopMarker)) return;

  // Track dependencies of the global memory load.
  llvm::SmallDenseSet<Operation*> asyncCopyDep;
  llvm::SmallDenseSet<Operation*> ldMatrixDep;

  // Count asyncCopyOp, ldMatrixOp, and mmaSyncOp
  int numAsyncCopyOp{0}, numLdMatrixOp{0}, numMmmaSyncOp{0};

  for (Operation& op : forOp.getBody()->getOperations()) {
    
    if (isa<nvgpu::DeviceAsyncCopyOp>(op)) numAsyncCopyOp++;
    else if (isa<nvgpu::LdMatrixOp>(op)) numLdMatrixOp++;
    else if (isa<nvgpu::MmaSyncOp>(op)) numMmmaSyncOp++;

    if (op.hasAttr(kPipeliningGlobalLoad)) {
      addDepOps(asyncCopyDep, &op, forOp.getBody());
    }
    if (op.hasAttr(kPipeliningLdmatrix)) {
      addDepOps(ldMatrixDep, &op, forOp.getBody());
    }
  }

#if 0 // Check the instruction mix of the mainloop
  std::cout << "nvgpu::DeviceAsyncCopyOp : " << numAsyncCopyOp << std::endl
            << "nvgpu::LdMatrixOp        : " << numLdMatrixOp << std::endl
            << "nvgpu::MmaSyncOp         : " << numMmmaSyncOp << std::endl;
#endif

  // Create a modulo schedule with loads from global memory and the operations
  // it depends on in stage 0. Store to shared memory and computation are in
  // stage `maxDepth`. In order to have a correct scheduling even with back
  // edges we order stages in decreasing order. 
  
  
  // Course-grained scheduling software pipelines global-to-shared copy (async_copy), 
  // shared-to-register loads (ldmatrix), and math on register operands (mma.sync)

  // Schedule mma.sync (x128) + ldmatrix (x24)
  for (Operation& op : forOp.getBody()->getOperations()) {
    if (!asyncCopyDep.count(&op) && !ldMatrixDep.count(&op) &&
        !isa<scf::YieldOp>(op)) {
      ops.push_back(std::make_pair(&op, depth));
    }
  }

  // Schedule async_copy (x16)
  for (Operation& op : forOp.getBody()->getOperations()) {
    if (asyncCopyDep.count(&op)) ops.push_back(std::make_pair(&op, 0));
  } 

  // Schedule async_cp_wait 2, barrier.sync 0, and ldmatrix (x8)
  for (Operation& op : forOp.getBody()->getOperations()) {
    if (ldMatrixDep.count(&op) && !asyncCopyDep.count(&op))
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
 // asyncCopyOp.getDstElements().getZExtValue()
  Value dstElements = 
      rewriter.create<arith::ConstantOp>(loc, asyncCopyOp.getDstElementsAttr());

  Value c0Index =
      rewriter.create<arith::ConstantIndexOp>(loc, 0);

  auto srcElements = rewriter.create<arith::SelectOp>(loc, pred, dstElements, c0Index);

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

struct KgroupOperations {
  llvm::SetVector<Operation*> ldMatrixLhsOps; // OperandA ldmatrixOps
  llvm::SetVector<Operation*> ldMatrixRhsOps; // OperandB ldmatrixOps
  llvm::SetVector<Operation*> mmaSyncOps;     // mmaSyncOps 
};


namespace {
struct GPUPipeliningPass : public GPUPipeliningBase<GPUPipeliningPass> {
  
  // Obtain using static tile sizes and instructions shapes
  static int const numLdMatrixOpPerOperandPerKblock = 4;
  static int const numMmaOpsPerKblock = 32;

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
      //int ldmatrixCounter = 0;           // OperandA
      //int ldmatrixTransposeCounter = 0;  // OperandB


      // Pipeline LdsmOp for kgroup = 0 (first 32 mmaOps)
      llvm::SetVector<Operation*> lhsLdsmSeqKgroup0, rhsLdsmSeqKgroup0;
      int numMmaOp = 0;
      for (Operation& op : forOp.getBody()->getOperations()) {
        if (auto mmaOp = dyn_cast<nvgpu::MmaSyncOp>(op)) {
          numMmaOp++;
          if (numMmaOp == 32) break;
          getMmaSyncLoadSequence(mmaOp, lhsLdsmSeqKgroup0, rhsLdsmSeqKgroup0);
        }
      }

      // std::cout << "Lhs Seq " << std::endl;
      for (auto op : lhsLdsmSeqKgroup0) {
        //op->dump();
        op->setAttr(kPipeliningLdmatrix, builder.getUnitAttr());
      }

      // std::cout << "Rhs Seq " << std::endl;
      for (auto op : rhsLdsmSeqKgroup0) {
        //op->dump();
        op->setAttr(kPipeliningLdmatrix, builder.getUnitAttr());
      }

      // Pipeline AsyncCopyOp
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

#if 0 // DO NOT USE this part
      // Use kgroup tracking for kgroup = 0 to pipeline ldsm
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
#endif
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

    // Post-processing fine-grained scheduling.
    // Interleaving ldmatrix and mma.sync
    funcOp.walk([](scf::ForOp forOp) {

      
    llvm::SetVector<Operation*> ldMatrixLhsOps;
    llvm::SetVector<Operation*> ldMatrixRhsOps; 
    llvm::SetVector<Operation*> mmaSyncOps; 
    Operation* beforeOp;
    
    std::vector<KgroupOperations> kgroupOperations;

    int numMmaOp = 0;
    int kgroup = 0;
    kgroupOperations.push_back(KgroupOperations());

    for (Operation& op : forOp.getBody()->getOperations()) {
      if (auto mmaOp = dyn_cast<nvgpu::MmaSyncOp>(op)) {
        numMmaOp++;
        kgroupOperations[kgroup].mmaSyncOps.insert(&op);
        getMmaSyncLoadSequence(mmaOp, kgroupOperations[kgroup].ldMatrixLhsOps, kgroupOperations[kgroup].ldMatrixRhsOps);
      }

      if (numMmaOp == numMmaOpsPerKblock) { // collected mmaSyncOp for one kgroup

        kgroupOperations.push_back(KgroupOperations());
        //std::cout << kgroupOperations[kgroup].mmaSyncOps.size() << std::endl;
        //std::cout << kgroupOperations[kgroup].ldMatrixLhsOps.size() << std::endl;
        //std::cout << kgroupOperations[kgroup].ldMatrixRhsOps.size() << std::endl;

        // Reset for the next kgroup
        numMmaOp = 0;
        kgroup++;

      }

      if (isa<nvgpu::DeviceAsyncWaitOp>(op)) {
        beforeOp = &op; 
        break;
      }
    }

    // Schedule kgroupOperations[0, 1, ..., ]
    int totalKgroups = kgroupOperations.size();
    for (int kgroup = 0; kgroup < totalKgroups - 1; kgroup++) {

      // Move loads for OperandA
      for (auto op : kgroupOperations[kgroup + 1].ldMatrixLhsOps) {
        
        std::vector<Operation*> userOps;
        for (Operation *userOp : op->getUsers()) {
          if (isa<vector::ExtractStridedSliceOp>(userOp)) {
            userOps.push_back(userOp);
          }
        }
        moveOperationsBefore(op, userOps, beforeOp);
      }

      // Move loads for OperandB
      for (auto op : kgroupOperations[kgroup + 1].ldMatrixRhsOps) {
        
        std::vector<Operation*> userOps;
        for (Operation *userOp : op->getUsers()) {
          if (isa<vector::ExtractStridedSliceOp>(userOp)) {
            userOps.push_back(userOp);
          }
        }
        moveOperationsBefore(op, userOps, beforeOp);
      }

      // Move math ops
      for (auto op : kgroupOperations[kgroup].mmaSyncOps) {
        std::vector<Operation*> userOps;
        for (Operation *userOp : op->getUsers()) {
          if (isa<vector::InsertStridedSliceOp>(userOp)) {
            userOps.push_back(userOp);
          }
        }
        moveOperationsBefore(op, userOps, beforeOp);
      }
    }

    // Move math ops (last mma.sync op)
    for (auto op : kgroupOperations[totalKgroups - 1].mmaSyncOps) {
      std::vector<Operation*> userOps;
      for (Operation *userOp : op->getUsers()) {
        if (isa<vector::InsertStridedSliceOp>(userOp)) {
          userOps.push_back(userOp);
        }
      }
      moveOperationsBefore(op, userOps, beforeOp);
    }

    });

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
