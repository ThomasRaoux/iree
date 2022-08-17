// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/KernelConfig.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Support/MathExtras.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/SideEffectUtils.h"

#define DEBUG_TYPE "iree-llvmgpu-distribute"

namespace mlir {
namespace iree_compiler {

// TODO: Maybe we need both a transform.iree.cpu.bufferize and a
// transform.iree.gpu.bufferize rather than a single common bufferize op?

/// Apply the permutation `perm` to `vals.
/// Return failure if perm is not a permutation.
// TODO: upstream as extraClassDeclaration once stabilized.
template <typename T>
static FailureOr<SmallVector<T>> permute(const SmallVector<T> &vals,
                                         ArrayRef<int64_t> perm) {
  if (vals.size() != perm.size()) return failure();
  SmallVector<T> result(vals.size());
  SmallVector<bool> seen(vals.size());
  for (const auto &it : llvm::zip(perm, vals)) {
    // Already seen, invalid thread_dim_mapping.
    if (seen[std::get<0>(it)]) return failure();
    result[std::get<0>(it)] = std::get<1>(it);
    seen[std::get<0>(it)] = true;
  }
  // Some not seen, invalid thread_dim_mapping.
  if (!llvm::all_of(seen, [](bool b) { return b; })) return failure();
  return result;
}

/// Helper to get apply the `thread_dim_mapping` permutation of a
/// `foreachThreadOp` to `values`.
// TODO: upstream as extraClassDeclaration once stabilized.
template <typename T>
static FailureOr<SmallVector<T>> getPermuted(
    scf::ForeachThreadOp foreachThreadOp, const SmallVector<T> &values) {
  // Apply mapping permutation if specified.
  auto mapping = foreachThreadOp.getThreadDimMapping();
  if (mapping && !mapping.empty()) {
    auto maybePermuted = permute(values, extractFromI64ArrayAttr(mapping));
    if (failed(maybePermuted))
      return foreachThreadOp->emitError("invalid permutation");
    return *maybePermuted;
  }
  return values;
}

/// Helper to get the `num_threads` of a `foreachThreadOp` after applying the
/// `thread_dim_mapping` permutation.
// TODO: upstream as extraClassDeclaration once stabilized.
static FailureOr<SmallVector<OpFoldResult>> getNumThreads(
    OpBuilder &b, scf::ForeachThreadOp foreachThreadOp) {
  SmallVector<OpFoldResult> threadCount = foreachThreadOp.getNumThreads();
  return getPermuted(foreachThreadOp, threadCount);
}

/// Helper to get the thread indices of a `foreachThreadOp` after applying the
/// `thread_dim_mapping` permutation.
// TODO: upstream as extraClassDeclaration once stabilized.
static FailureOr<SmallVector<Value>> getThreadIndices(
    OpBuilder &b, scf::ForeachThreadOp foreachThreadOp) {
  SmallVector<Value> threadCount = foreachThreadOp.getThreadIndices();
  return getPermuted(foreachThreadOp, threadCount);
}

static LogicalResult rewriteForeachThreadToGpu(
    scf::ForeachThreadOp foreachThreadOp,
    const SmallVector<int64_t> &globalWorkgroupSizes, RewriterBase &rewriter) {
  if (foreachThreadOp.getNumResults() > 0)
    return foreachThreadOp->emitError(
        "only bufferized scf.foreach_thread lowers to gpu.thread");

  auto maybeWorkgroupSizes = getNumThreads(rewriter, foreachThreadOp);
  if (failed(maybeWorkgroupSizes))
    return foreachThreadOp->emitError("unsupported dynamic workgroup size");

  SmallVector<OpFoldResult> dynamicWorkgroupSize = *maybeWorkgroupSizes;

  // Step 1. Create the gpu.thread ops
  Location loc = foreachThreadOp.getLoc();
  IndexType indexType = rewriter.getIndexType();

  SmallVector<gpu::Dimension, 3> gpuDims{gpu::Dimension::x, gpu::Dimension::y,
                                         gpu::Dimension::z};
  SmallVector<Value, 3> threadOps;
  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  // Step 2. Maybe create conditionals to predicate the region.
  int64_t idx = 0;
  Value predicate;
  for (OpFoldResult workgroupSize : dynamicWorkgroupSize) {
    auto cstWgSize = getConstantIntValue(workgroupSize);
    // Skip dimensions not distributed.
    if(cstWgSize && *cstWgSize == 1) {
      threadOps.push_back(zero);
      continue;
    }
    if(idx >= 3)
      return foreachThreadOp->emitError("more than 3 distributed dims.");
    int64_t globalWorkgroupSize = globalWorkgroupSizes[idx];
    Value threadId =
        rewriter.create<gpu::ThreadIdOp>(loc, indexType, gpuDims[idx]);
    threadOps.push_back(threadId);
    idx++;
    if (cstWgSize) {
      assert(*cstWgSize <= globalWorkgroupSize && "workgroup size overflow");
      if (*cstWgSize == globalWorkgroupSize) continue;
    }
    Value tmpPredicate = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ult, threadId,
        linalg::materializeOpFoldResult(rewriter, loc, workgroupSize));
    predicate =
        predicate ? rewriter.create<arith::AndIOp>(loc, predicate, tmpPredicate)
                  : tmpPredicate;
  }

  // Step 3. Move the body of foreachThreadOp.
  // Erase the terminator first, it will not be used.
  rewriter.eraseOp(foreachThreadOp.getTerminator());
  Block *targetBlock;
  Block::iterator insertionPoint;
  if (predicate) {
    // Step 3.a. If predicated, move at the beginning.
    auto ifOp =
        rewriter.create<scf::IfOp>(loc, predicate, /*withElseRegion=*/false);
    targetBlock = ifOp.thenBlock();
    insertionPoint = ifOp.thenBlock()->begin();
  } else {
    // Step 3.a. Otherwise, move inline just before foreachThreadOp.
    targetBlock = foreachThreadOp->getBlock();
    insertionPoint = Block::iterator(foreachThreadOp);
  }
  Block &sourceBlock = foreachThreadOp.getRegion().front();
  targetBlock->getOperations().splice(insertionPoint,
                                      sourceBlock.getOperations());

  // Step 4. RAUW thread indices to thread ops.
  SmallVector<Value> threadIndices =
      *getThreadIndices(rewriter, foreachThreadOp);
  for (auto it : llvm::zip(threadIndices, threadOps)) {
    if (!std::get<0>(it)) continue;
    std::get<0>(it).replaceAllUsesWith(std::get<1>(it));
  }

  // Step 6. Erase old op.
  rewriter.eraseOp(foreachThreadOp);

  return success();
}

namespace {
struct LLVMGPUDistributePass
    : public LLVMGPUDistributeBase<LLVMGPUDistributePass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, gpu::GPUDialect>();
  }
  void runOnOperation() override {
    auto funcOp = getOperation();
    if (!isEntryPoint(funcOp)) return;

    auto workgroupSize = llvm::to_vector(llvm::map_range(
        getEntryPoint(funcOp)->getWorkgroupSize().getValue(),
        [&](Attribute attr) { return attr.cast<IntegerAttr>().getInt(); }));
    SmallVector<scf::ForeachThreadOp> foreachOps;
    funcOp.walk(
        [&](scf::ForeachThreadOp foreachOp) { foreachOps.push_back(foreachOp); });
    for (scf::ForeachThreadOp op : foreachOps) {
      IRRewriter rewriter(op->getContext());
      rewriter.setInsertionPoint(op);
      if(failed(rewriteForeachThreadToGpu(op, workgroupSize, rewriter))) {
        return signalPassFailure();
      }
    }
  }
};
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createLLVMGPUDistribute() {
  return std::make_unique<LLVMGPUDistributePass>();
}

}  // namespace iree_compiler
}  // namespace mlir
