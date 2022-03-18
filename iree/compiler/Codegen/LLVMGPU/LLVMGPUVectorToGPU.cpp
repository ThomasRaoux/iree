// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "llvm/ADT/SetVector.h"
#include "mlir/Conversion/VectorToGPU/VectorToGPU.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {

/// Helper to convert copy to shared memory to async copy. This creates groups
/// of consecutive copies and emit wait operation right after.
static void createAsyncGroups(FuncOp funcOp) {
  llvm::SmallSetVector<vector::TransferWriteOp, 16> copyToSharedMem;
  // Look for all the copy that can be converted to async copy ops.
  funcOp.walk([&](vector::TransferWriteOp writeOp) {
    if (!writeOp.permutation_map().isMinorIdentity() ||
        writeOp.getVectorType().getRank() != 1 || !writeOp.isDimInBounds(0) ||
        writeOp.getShapedType().cast<MemRefType>().getMemorySpaceAsInt() !=
            gpu::GPUDialect::getWorkgroupAddressSpace())
      return WalkResult::advance();
    auto read = writeOp.vector().getDefiningOp<vector::TransferReadOp>();
    if (!read || read.getVectorType() != writeOp.getVectorType() ||
        !read.isDimInBounds(0) || !read.permutation_map().isMinorIdentity())
      return WalkResult::advance();
    if (read.getVectorType().getNumElements() > 4 ||
        !read.getVectorType().getElementType().isF32())
      return WalkResult::advance();
    copyToSharedMem.insert(writeOp);
    return WalkResult::advance();
  });

  while (!copyToSharedMem.empty()) {
    SmallVector<vector::TransferWriteOp> group;
    vector::TransferWriteOp writeOp = *copyToSharedMem.begin();
    // Start a group with the first write.
    copyToSharedMem.remove(writeOp);
    group.push_back(writeOp);
    Operation* nextNode = writeOp.getOperation();
    // Look in the next nodes for more copies to add to the same group.
    while ((nextNode = nextNode->getNextNode())) {
      // Ignore ops without side effects
      auto memInterface = dyn_cast<MemoryEffectOpInterface>(nextNode);
      if (memInterface && memInterface.hasNoEffect() &&
          !nextNode->hasTrait<OpTrait::HasRecursiveSideEffects>())
        continue;
      auto readOp = dyn_cast<vector::TransferReadOp>(nextNode);
      // ignore read from a different address space.
      if (readOp &&
          readOp.getShapedType().cast<MemRefType>().getMemorySpaceAsInt() !=
              gpu::GPUDialect::getWorkgroupAddressSpace()) {
        continue;
      }
      auto nextWriteOp = dyn_cast<vector::TransferWriteOp>(nextNode);
      if (nextWriteOp && copyToSharedMem.count(nextWriteOp)) {
        // found another copy, add it to the group.
        copyToSharedMem.remove(nextWriteOp);
        group.push_back(nextWriteOp);
        continue;
      }
      // If the op is something else stop the accumulating op in the group.
      break;
    }
    // emit the group.
    SmallVector<Value> tokens;
    OpBuilder builder(funcOp.getContext());
    for (vector::TransferWriteOp writeOp : group) {
      builder.setInsertionPoint(writeOp);
      auto readOp = writeOp.vector().getDefiningOp<vector::TransferReadOp>();
      Value token = builder.create<gpu::DeviceAsyncCopyOp>(
          writeOp.getLoc(), gpu::DeviceAsyncTokenType::get(funcOp.getContext()),
          writeOp.source(), writeOp.indices(), readOp.source(),
          readOp.indices(),
          builder.getIndexAttr(readOp.getVectorType().getNumElements()));
      tokens.push_back(token);
    }
    // Create the group and wait for it right after.
    Value groupToken = builder.create<gpu::DeviceAsyncCreateGroupOp>(
        funcOp.getLoc(), gpu::DeviceAsyncTokenType::get(funcOp.getContext()),
        tokens);
    builder.create<gpu::DeviceAsyncWaitOp>(funcOp.getLoc(), groupToken,
                                           nullptr);
    // Clean up old stores.
    for (vector::TransferWriteOp writeOp : group) writeOp.erase();
  }
}

static MemRefType dropUnitDims(MemRefType inputType, ArrayRef<int64_t> offsets,
                               ArrayRef<int64_t> sizes,
                               ArrayRef<int64_t> strides) {
  Type rankReducedType = memref::SubViewOp::inferRankReducedResultType(
      0, inputType, offsets, sizes, strides);
  return canonicalizeStridedLayout(rankReducedType.cast<MemRefType>());
}

static Value rankReducingSubviewDroppingUnitDims(PatternRewriter &rewriter,
                                                 mlir::Location loc,
                                                 Value input) {
  MemRefType inputType = input.getType().cast<MemRefType>();
  assert(inputType.hasStaticShape());
  SmallVector<int64_t> subViewOffsets(inputType.getRank(), 0);
  SmallVector<int64_t> subViewStrides(inputType.getRank(), 1);
  ArrayRef<int64_t> subViewSizes = inputType.getShape();
  MemRefType resultType =
      dropUnitDims(inputType, subViewOffsets, subViewSizes, subViewStrides);
  if (canonicalizeStridedLayout(resultType) ==
      canonicalizeStridedLayout(inputType))
    return input;
  return rewriter.create<memref::SubViewOp>(
      loc, resultType, input, subViewOffsets, subViewSizes, subViewStrides);
}

static Value collapseContiguousRowMajorMemRefTo2D(PatternRewriter &rewriter,
                                                  mlir::Location loc,
                                                  Value input) {
  Value rankReducedInput =
      rankReducingSubviewDroppingUnitDims(rewriter, loc, input);
  ShapedType rankReducedInputType =
      rankReducedInput.getType().cast<ShapedType>();
  if (rankReducedInputType.getRank() == 1) return rankReducedInput;
  llvm::SmallVector<ReassociationIndices> indicesArray;
  ReassociationIndices collapasedIndices;
  for (int i = 0; i < rankReducedInputType.getRank() - 1; ++i)
    collapasedIndices.push_back(i);
  indicesArray.push_back(collapasedIndices);
  indicesArray.push_back({rankReducedInputType.getRank() - 1});
  return rewriter.create<memref::CollapseShapeOp>(loc, rankReducedInput,
                                                  indicesArray);
}

namespace {

struct FlattenTransferReadOp : public OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern<vector::TransferReadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferReadOp transferReadOp,
                                PatternRewriter &rewriter) const override {
    auto loc = transferReadOp.getLoc();
    Value vector = transferReadOp.vector();
    VectorType vectorType = vector.getType().cast<VectorType>();
    Value source = transferReadOp.source();
    MemRefType sourceType = source.getType().dyn_cast<MemRefType>();
    // Contiguity check is valid on tensors only.
    if (!sourceType) return failure();
    // Pattern only supported for 3D where second dim is unit
    if (vectorType.getRank() != 3 || vectorType.getShape()[1] != 1)
      return failure();
    // TODO: generalize this pattern, relax the requirements here.
    if (transferReadOp.hasOutOfBoundsDim()) return failure();
    if (!transferReadOp.permutation_map().isMinorIdentity()) return failure();
    if (transferReadOp.mask()) return failure();
    MemRefType sourceType2D =
        MemRefType::get({sourceType.getShape()[0], sourceType.getShape()[1]},
                        sourceType.getElementType());
    AffineMap linearLayoutMap = getStridedLinearLayoutMap(sourceType2D);
    // TODO (nirvedhmeshram): add support for dynamic shapes
    if (!linearLayoutMap || linearLayoutMap.getNumSymbols() != 0)
      return failure();
    ValueRange indices = transferReadOp.indices();
    Value linearindex = makeComposedAffineApply(rewriter, loc, linearLayoutMap,
                                                {indices[0], indices[1]});
    ArrayAttr newInBoundsAttr =
        rewriter.getBoolArrayAttr(SmallVector<bool>(2, true));
    auto identityMap1D = rewriter.getMultiDimIdentityMap(2);
    VectorType vectorType2d =
        VectorType::get({vectorType.getShape()[0] * vectorType.getShape()[1],
                         vectorType.getShape()[2]},
                        vectorType.getElementType());
    VectorType vectorType3d =
        VectorType::get({vectorType.getShape()[1], vectorType.getShape()[0],
                         vectorType.getShape()[2]},
                        vectorType.getElementType());
    Value source2d =
        collapseContiguousRowMajorMemRefTo2D(rewriter, loc, source);
    Value read2d = rewriter.create<vector::TransferReadOp>(
        loc, vectorType2d, source2d, ValueRange{linearindex, indices[2]},
        identityMap1D, transferReadOp.padding(), transferReadOp.mask(),
        newInBoundsAttr);
    Value read3d =
        rewriter.create<vector::BroadcastOp>(loc, vectorType3d, read2d);
    static constexpr std::array<int64_t, 3> perm = {1, 0, 2};
    rewriter.replaceOpWithNewOp<vector::TransposeOp>(transferReadOp, read3d,
                                                     perm);
    return success();
  }
};

struct LLVMGPUVectorToGPUPass
    : public LLVMGPUVectorToGPUBase<LLVMGPUVectorToGPUPass> {
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<gpu::GPUDialect, AffineDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    auto funcOp = getOperation();
    RewritePatternSet flatternpatterns(funcOp.getContext());
    flatternpatterns.insert<FlattenTransferReadOp>(funcOp.getContext());
    if (failed(applyPatternsAndFoldGreedily(funcOp,
                                            std::move(flatternpatterns)))) {
      return signalPassFailure();
    }
    RewritePatternSet patterns(funcOp.getContext());
    populatePrepareVectorToMMAPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));

    convertVectorToMMAOps(funcOp);
    createAsyncGroups(funcOp);
  }
};
}  // namespace

std::unique_ptr<OperationPass<FuncOp>> createLLVMGPUVectorToGPU() {
  return std::make_unique<LLVMGPUVectorToGPUPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
