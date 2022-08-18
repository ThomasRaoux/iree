// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree-dialects/Dialect/LinalgExt/Passes/Transforms.h"
#include "iree-dialects/Dialect/LinalgExt/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Dialect/LoweringConfig.h"
#include "iree/compiler/Codegen/LLVMGPU/KernelConfig.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Codegen/Utils/MarkerUtils.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Support/MathExtras.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include <numeric>

#define DEBUG_TYPE "iree-llvmgpu-tile-tensor"

namespace mlir {
namespace iree_compiler {

/// Patterns for workgroup level tiling. Workgroup tiling is done at the flow
/// level but we may have extra tiling for the reduction dimension. Therefore we
/// tile again without distributing.
static void populateTilingReductionPatterns(RewritePatternSet &patterns) {
  auto tileSizesFn = [&](OpBuilder &builder,
                         Operation *op) -> SmallVector<Value, 4> {
    auto interfaceOp = cast<PartitionableLoopsInterface>(*op);
    auto partitionedLoops =
        interfaceOp.getPartitionableLoops(kNumMaxParallelDims);
    SmallVector<Value, 4> tileSizes = getTileSizes(builder, op, 0);
    auto zero = builder.create<arith::ConstantIndexOp>(op->getLoc(), 0);
    for (unsigned depth : partitionedLoops) {
      if (depth < tileSizes.size()) {
        tileSizes[depth] = zero;
      }
    }
    return tileSizes;
  };

  auto tilingOptions = linalg::LinalgTilingOptions()
                           .setLoopType(linalg::LinalgTilingLoopType::Loops)
                           .setTileSizeComputationFunction(tileSizesFn);
  MLIRContext *context = patterns.getContext();

  linalg::LinalgTransformationFilter filter(
      ArrayRef<StringAttr>{
          StringAttr::get(context, getWorkgroupMemoryMarker())},
      StringAttr::get(context, getVectorizeMarker()));
  filter.setMatchByDefault();
  linalg::TilingPatterns<linalg::MatmulOp, linalg::BatchMatmulOp,
                         linalg::GenericOp>::insert(patterns, tilingOptions,
                                                    filter);
}

static LogicalResult tileReduction(func::FuncOp funcOp) {
  {
    // Tile again at the workgroup level since redution dimension were
    // ignored. Dimensions already tiled will be ignore since we tile to the
    // same size.
    RewritePatternSet wgTilingPatterns(funcOp.getContext());
    populateTilingReductionPatterns(wgTilingPatterns);
    if (failed(applyPatternsAndFoldGreedily(funcOp,
                                            std::move(wgTilingPatterns)))) {
      return failure();
    }
  }

  {
    RewritePatternSet wgTilingCanonicalizationPatterns =
        linalg::getLinalgTilingCanonicalizationPatterns(funcOp.getContext());
    populateAffineMinSCFCanonicalizationPattern(
        wgTilingCanonicalizationPatterns);
    if (failed(applyPatternsAndFoldGreedily(
            funcOp, std::move(wgTilingCanonicalizationPatterns)))) {
      return failure();
    }
    return success();
  }
}

/// Return the tile size associated to one thread or warp based on the number of
/// element in the group.
static SmallVector<Value, 4> calculateDistributedTileSize(
    ArrayRef<int64_t> numElements, OpBuilder &builder, Operation *operation) {
  SmallVector<int64_t> blockTileSize = getTileSizes(operation, 0);
  SmallVector<Value, 4> tileSizesVal;
  // Use partitionedLoop to know what loop needs to be distributed.
  auto interfaceOp = cast<PartitionableLoopsInterface>(*operation);
  auto partitionedLoops =
      interfaceOp.getPartitionableLoops(kNumMaxParallelDims);
  if (partitionedLoops.empty()) {
    return tileSizesVal;
  }
  unsigned maxDepth = partitionedLoops.back() + 1;
  auto zero = builder.create<arith::ConstantIndexOp>(operation->getLoc(), 0);
  tileSizesVal.resize(maxDepth, zero);
  // partitionedLoops contains the dimensions we want to distribute.
  // We are distributing them in order onto the different workgroup
  // dimensions.
  SmallVector<int64_t> distributedDim(numElements.begin(), numElements.end());
  distributedDim.resize(partitionedLoops.size());
  unsigned idIdx = 0;
  std::reverse(distributedDim.begin(), distributedDim.end());
  for (unsigned depth : partitionedLoops) {
    if (depth >= blockTileSize.size()) continue;
    tileSizesVal[depth] = builder.create<arith::ConstantIndexOp>(
        operation->getLoc(),
        llvm::divideCeil(blockTileSize[depth], distributedDim[idIdx++]));
    if (idIdx == kNumMaxParallelDims) break;
  }
  return tileSizesVal;
}

/// Patterns for warp level tiling.
static void populateTilingPatterns(
    RewritePatternSet &patterns, SmallVectorImpl<int64_t> &workgroupSize, bool toWarp) {
  std::array<int64_t, 3> workersPerWorkgroup = {
      toWarp ? workgroupSize[0] / kWarpSize : workgroupSize[0], workgroupSize[1], workgroupSize[2]};

  linalg::TileSizeComputationFunction getInnerTileSizeFn =
      [workersPerWorkgroup](OpBuilder &builder, Operation *operation) {
        return calculateDistributedTileSize(workersPerWorkgroup, builder,
                                            operation);
      };
  auto tilingOptions = linalg::LinalgTilingOptions()
                           .setLoopType(linalg::LinalgTilingLoopType::Loops)
                           .setTileSizeComputationFunction(getInnerTileSizeFn);
  MLIRContext *context = patterns.getContext();
  linalg::LinalgTransformationFilter f(
      {StringAttr::get(context, getWorkgroupKTiledMarker()),
       StringAttr::get(context, getWorkgroupMemoryMarker())},
      StringAttr::get(context, getVectorizeMarker()));
  f.addFilter([](Operation *op) {
     // FFT doesn't support second level of tiling yet.
     return success(!isa<IREE::LinalgExt::FftOp>(op));
   }).setMatchByDefault();
  patterns.insert<linalg::LinalgTilingPattern,
                  IREE::LinalgExt::TilingInterfaceTilingPattern>(
      context, tilingOptions, f);
}


static LogicalResult tileParallelDims(
    func::FuncOp funcOp, SmallVectorImpl<int64_t> &workgroupSize,
    bool distributeToWarp) {
  std::array<int64_t, 3> elementPerWorkgroup = {
      distributeToWarp ? workgroupSize[0] / kWarpSize : workgroupSize[0],
      workgroupSize[1], workgroupSize[2]};
  SmallVector<Operation *> computeOps;
  SmallVector<LoopTilingAndDistributionInfo> tiledLoops;
  if (failed(getComputeOps(funcOp, computeOps, tiledLoops))) {
    return funcOp.emitOpError("failed to get compute ops");
  }

  for (Operation *op : computeOps) {
    auto tilingOp = dyn_cast<TilingInterface>(op);
    if(!tilingOp)
      continue;
    size_t numLoops = 0;
    for(auto type : tilingOp.getLoopIteratorTypes()) {
      if(type == getParallelIteratorTypeName())
        numLoops++;
    }
    IRRewriter rewriter(op->getContext());
    rewriter.setInsertionPoint(op);
    auto interfaceOp = cast<PartitionableLoopsInterface>(*op);
    auto partitionedLoops =
        interfaceOp.getPartitionableLoops(kNumMaxParallelDims);
    SmallVector<OpFoldResult> numThreads(numLoops, rewriter.getIndexAttr(0));
    int64_t id = 0;
    int64_t threadId = 0;
    SmallVector<int64_t> idDims;
    for (unsigned loop : llvm::reverse(partitionedLoops)) {
      int64_t num = elementPerWorkgroup[id++];
      if (num > 1) {
        numThreads[loop] = rewriter.getIndexAttr(num);
        idDims.push_back(threadId++);
      }
    }
    std::reverse(idDims.begin(), idDims.end());
    for(int64_t i = threadId; i < 3; i++)
      idDims.push_back(i);
    auto tilingResult = linalg::tileToForeachThreadOp(
        rewriter, tilingOp, numThreads, idDims);
    rewriter.replaceOp(op, tilingResult->tileOp->getResults());
  }
  return success();
}

namespace {
struct LLVMGPUTileTensorPass
    : public LLVMGPUTileTensorBase<LLVMGPUTileTensorPass> {
 private:
  // Distribute the workloads to warp if true otherwise distribute to threads.
  bool distributeToWarp = false;

 public:
  LLVMGPUTileTensorPass(bool distributeToWarp)
      : distributeToWarp(distributeToWarp) {}
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, gpu::GPUDialect>();
  }
  void runOnOperation() override {
    auto funcOp = getOperation();
    if (!isEntryPoint(funcOp)) return;

    auto workgroupSize = llvm::to_vector<4>(llvm::map_range(
        getEntryPoint(funcOp)->getWorkgroupSize().getValue(),
        [&](Attribute attr) { return attr.cast<IntegerAttr>().getInt(); }));
    if (failed(tileParallelDims(funcOp, workgroupSize, distributeToWarp))) {
      return signalPassFailure();
    }

    LLVM_DEBUG({
      llvm::dbgs() << "--- After second level of tiling";
      funcOp.dump();
    });

    if (failed(tileReduction(funcOp))) {
      return signalPassFailure();
    }

    LLVM_DEBUG({
      llvm::dbgs() << "--- After tile reductions:";
      funcOp.dump();
    });
  }
};
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createLLVMGPUTileTensor(
    bool distributeToWarp) {
  return std::make_unique<LLVMGPUTileTensorPass>(distributeToWarp);
}

}  // namespace iree_compiler
}  // namespace mlir
