// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "iree/compiler/Conversion/CodegenUtils/FunctionUtils.h"
#include "iree/compiler/Conversion/CodegenUtils/MarkerUtils.h"
#include "iree/compiler/Conversion/Common/Transforms.h"
#include "iree/compiler/Conversion/LinalgToNVVM/KernelConfig.h"
#include "iree/compiler/Conversion/LinalgToNVVM/Passes.h"
#include "iree/compiler/Dialect/IREE/IR/IREEOps.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/IR/Matchers.h"

namespace mlir {
namespace iree_compiler {
static constexpr int32_t kNumGPUDims = 3;
static constexpr unsigned kWorkgroupMemoryAddressSpace = 3;

static SmallVector<linalg::ProcInfo, 2> getGPUThreadIdsAndCounts(
    OpBuilder &builder, Location loc, unsigned numDims,
    ArrayRef<int64_t> workgroupSize) {
  assert(numDims <= kNumGPUDims);
  SmallVector<linalg::ProcInfo, 2> procInfo(numDims);
  std::array<StringRef, kNumGPUDims> dimAttr{"x", "y", "z"};
  Type indexType = builder.getIndexType();
  for (unsigned i = 0; i < numDims; ++i) {
    StringAttr attr = builder.getStringAttr(dimAttr[i]);
    procInfo[numDims - 1 - i] = {
        builder.create<gpu::ThreadIdOp>(loc, indexType, attr),
        builder.create<ConstantOp>(loc,
                                   builder.getIndexAttr(workgroupSize[i]))};
  }
  return procInfo;
}

/// Patterns for thread level tiling.
static void populateTilingReductionPatterns(
    MLIRContext *context, OwningRewritePatternList &patterns,
    ArrayRef<int64_t> tileSizes) {

  auto tilingOptions = linalg::LinalgTilingOptions()
                           .setLoopType(linalg::LinalgTilingLoopType::Loops)
                           .setTileSizes(tileSizes);

  patterns.insert<linalg::LinalgTilingPattern<linalg::MatmulOp>,
                  linalg::LinalgTilingPattern<linalg::BatchMatmulOp>>(
      context, tilingOptions,
      linalg::LinalgTransformationFilter(
          {Identifier::get(getWorkgroupMarker(), context)},
          Identifier::get(getWorkgroupKTiledMarker(), context)));
}

/// Patterns for thread level tiling.
static void populateTilingToInvocationPatterns(
    MLIRContext *context, OwningRewritePatternList &patterns,
    ArrayRef<int64_t> tileSizes, ArrayRef<int64_t> workgroupSize) {
  linalg::TileSizeComputationFunction getInnerTileSizeFn =
      [tileSizes](OpBuilder &builder, Operation *operation) {
        if (tileSizes.empty()) return SmallVector<Value, 4>();
        SmallVector<Value, 4> tileSizesVal;
        tileSizesVal.reserve(tileSizes.size());
        for (auto val : llvm::enumerate(tileSizes)) {
          // Only tile the last 3 dimensions. Use tile size of 0 for any higher
          // dimension as we only support distributing on 3 dimensions.
          int64_t t =
              (tileSizes.size() - val.index()) <= kNumGPUDims ? val.value() : 0;
          tileSizesVal.push_back(
              builder.create<ConstantIndexOp>(operation->getLoc(), t));
        }
        return tileSizesVal;
      };

  auto getThreadProcInfoFn = [workgroupSize](
                                 OpBuilder &builder, Location loc,
                                 ArrayRef<Range> parallelLoopRanges) {
    return getGPUThreadIdsAndCounts(builder, loc, parallelLoopRanges.size(),
                                    workgroupSize);
  };
  linalg::LinalgLoopDistributionOptions invocationDistributionOptions = {
      getThreadProcInfoFn,
      {linalg::DistributionMethod::Cyclic, linalg::DistributionMethod::Cyclic,
       linalg::DistributionMethod::Cyclic}};

  auto tilingOptions =
      linalg::LinalgTilingOptions()
          .setLoopType(linalg::LinalgTilingLoopType::Loops)
          .setTileSizeComputationFunction(getInnerTileSizeFn)
          .setDistributionOptions(invocationDistributionOptions);

  patterns.insert<linalg::LinalgTilingPattern<linalg::MatmulOp>,
                  linalg::LinalgTilingPattern<linalg::FillOp>,
                  linalg::LinalgTilingPattern<linalg::BatchMatmulOp>,
                  linalg::LinalgTilingPattern<linalg::GenericOp>>(
      context, tilingOptions,
      linalg::LinalgTransformationFilter(
          {Identifier::get(getWorkgroupMarker(), context),
           Identifier::get(getWorkgroupKTiledMarker(), context),
           Identifier::get(getWorkgroupMemoryMarker(), context)},
          Identifier::get(getVectorizeMarker(), context)));
}


static LogicalResult copyToWorkgroupMemory(OpBuilder &b, Value src, Value dst) {
  auto copyOp = b.create<linalg::CopyOp>(src.getLoc(), src, dst);
  setMarker(copyOp, getCopyToWorkgroupMemoryMarker());
  return success();
}
static StringRef getNumWorkgroupAttributionsAttrName() {
  return "workgroup_attributions";
}

static Optional<Value> allocateWorkgroupMemory(
    OpBuilder &b, memref::SubViewOp subview,
    ArrayRef<Value> boundingSubViewSize, OperationFolder *folder) {
  // Allocate the memory into the entry block of the parent FuncOp. This better
  // aligns with the semantics of this memory which is available at the entry of
  // the function.
  OpBuilder::InsertionGuard guard(b);
  FuncOp funcOp = subview->getParentOfType<FuncOp>();
  if (!funcOp) {
    subview.emitError("expected op to be within std.func");
    return llvm::None;
  }
  ModuleOp moduleOp = funcOp->getParentOfType<ModuleOp>();
  SymbolTable symbolTable(moduleOp);
  
  // The bounding subview size is expected to be constant. This specified the
  // shape of the allocation.
  SmallVector<int64_t, 2> shape = llvm::to_vector<2>(
      llvm::map_range(boundingSubViewSize, [](Value v) -> int64_t {
        APInt value;
        if (matchPattern(v, m_ConstantInt(&value))) return value.getSExtValue();
        return -1;
      }));
  if (llvm::any_of(shape, [](int64_t v) { return v == -1; })) return {};
  Type allocType =
      MemRefType::get(shape, subview.getType().getElementType(), {},
                      kWorkgroupMemoryAddressSpace);
  b.setInsertionPoint(&moduleOp.front());
  auto global = b.create<memref::GlobalOp>(
      funcOp.getLoc(), "__shared_memory__",
      /*sym_visibility=*/b.getStringAttr("private"),
      /*type=*/allocType,
      /*initial_value=*/ElementsAttr(),
      /*constant=*/false);
  symbolTable.insert(global);

  b.setInsertionPointToStart(&(*funcOp.getBody().begin()));
  Value buffer = b.create<memref::GetGlobalOp>(funcOp.getLoc(), global.type(),
                                               global.getName());
  return buffer;
}

static LogicalResult deallocateWorkgroupMemory(OpBuilder &b, Value buffer) {
  // Nothing to do.
  return success();
}

static void populatePromotionPatterns(MLIRContext *context,
                                      OwningRewritePatternList &patterns) {
  patterns.insert<linalg::LinalgPromotionPattern<linalg::MatmulOp>>(
      context,
      linalg::LinalgPromotionOptions()
          .setAllocationDeallocationFns(allocateWorkgroupMemory,
                                        deallocateWorkgroupMemory)
          .setCopyInOutFns(copyToWorkgroupMemory, copyToWorkgroupMemory)
          .setOperandsToPromote({0, 1})
          .setUseFullTileBuffers({false, false}),
      linalg::LinalgTransformationFilter(
          {Identifier::get(getWorkgroupKTiledMarker(), context)},
          Identifier::get(getWorkgroupMemoryMarker(), context)));
}

static constexpr unsigned kWorkgroupDimCount = 3;

namespace {

/// Replaces hal.interface.workgroup.size op with the constant value chosen
/// from tiling scheme.
class ConcretizeWorkgroupSizeOp final
    : public OpRewritePattern<IREE::HAL::InterfaceWorkgroupSizeOp> {
 public:
  ConcretizeWorkgroupSizeOp(MLIRContext *context, ArrayRef<int64_t> tileSize)
      : OpRewritePattern(context, /*benefit=*/1), tileSize(tileSize) {}

  LogicalResult matchAndRewrite(IREE::HAL::InterfaceWorkgroupSizeOp op,
                                PatternRewriter &rewriter) const override {
    unsigned dimIndex = op.dimension().getZExtValue();

    if (dimIndex < kWorkgroupDimCount && tileSize[dimIndex] != 0) {
      rewriter.replaceOpWithNewOp<ConstantOp>(
          op, rewriter.getIndexAttr(tileSize[dimIndex]));
      return success();
    }

    return failure();
  }

 private:
  ArrayRef<int64_t> tileSize;
};

struct TileAndDistributeToThreads
    : public PassWrapper<TileAndDistributeToThreads,
                         OperationPass<IREE::HAL::ExecutableTargetOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, gpu::GPUDialect>();
  }
  void runOnOperation() override {
    IREE::HAL::ExecutableTargetOp targetOp = getOperation();
    ModuleOp module = targetOp.getInnerModule();

    MLIRContext *context = module->getContext();
    for (FuncOp funcOp : module.getOps<FuncOp>()) {
      if (!isEntryPoint(funcOp)) continue;

      SmallVector<linalg::LinalgOp, 4> linalgOps;
      SmallVector<Operation *, 4> tiledLoops;

      if (failed(getLinalgOps(funcOp, linalgOps, tiledLoops))) {
        return signalPassFailure();
      }
      linalg::Aliases aliases;
      linalg::LinalgDependenceGraph dependenceGraph(aliases, linalgOps);
      auto config = getCUDALaunchConfig(context, dependenceGraph, linalgOps);
      if (!config) return signalPassFailure();

      // Attach the workgroup size as an attribute. This will be used when
      // creating the flatbuffer.
      funcOp->setAttr("cuda_workgroup_size",
                      DenseElementsAttr::get<int64_t>(
                          VectorType::get(3, IntegerType::get(context, 64)),
                          config->getWorkgroupSize()));

      Operation *rootOp = config->getRootOperation(llvm::to_vector<4>(
          llvm::map_range(linalgOps, [](linalg::LinalgOp op) {
            return op.getOperation();
          })));
      SmallVector<int64_t, 4> wgTileSize =
          llvm::to_vector<4>(config->getTileSizes(rootOp, 0));
      // If there is no tile size, skip tiling.
      if (wgTileSize.empty()) return;
      unsigned numOuterParallelLoops =
          getNumOuterParallelLoops(cast<linalg::LinalgOp>(rootOp));
      size_t numContractionLoops =
          wgTileSize.size() > numOuterParallelLoops
              ? wgTileSize.size() - numOuterParallelLoops
              : 0;
      size_t numTilableDims =
          std::min(kWorkgroupDimCount, numOuterParallelLoops);
      wgTileSize.resize(numTilableDims);
      std::reverse(wgTileSize.begin(), wgTileSize.end());
      {
        // Replace the opaque tile size for workgroup level tiling and update
        // the number of workgroups based on the tile size.
        OwningRewritePatternList patterns(context);
        patterns.insert<ConcretizeWorkgroupSizeOp>(context, wgTileSize);

        (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
        if (failed(materializeStaticLaunchInformation(funcOp, wgTileSize))) {
          funcOp.emitOpError("failed to materialize static launch information");
          return signalPassFailure();
        }
      }

      SmallVector<int64_t, 4> threadTileSize =
          llvm::to_vector<4>(config->getTileSizes(rootOp, 2));
      if(numContractionLoops > 0) {
        // Tile again at the workgroup level since redution dimension were
        // ignored. Dimensions already tiled will be ignore since we tile to the
        // same size.
        OwningRewritePatternList wgTilingPatterns(context);
        populateTilingReductionPatterns(context, wgTilingPatterns,
                                        config->getTileSizes(rootOp, 0));
        (void)applyPatternsAndFoldGreedily(funcOp, std::move(wgTilingPatterns));
        applyCanonicalizationPatternsForTiling(context, funcOp);
      }

      {
        OwningRewritePatternList promotionPatterns(&getContext());
        populatePromotionPatterns(context, promotionPatterns);
        (void)applyPatternsAndFoldGreedily(funcOp,
                                           std::move(promotionPatterns));
        applyCanonicalizationPatternsForTiling(context, funcOp);
      }

      {
        // Apply last level of tiling and distribute to threads.
        OwningRewritePatternList threadLevelTilingPatterns(context);
        populateTilingToInvocationPatterns(context, threadLevelTilingPatterns,
                                           threadTileSize,
                                           config->getWorkgroupSize());
        (void)applyPatternsAndFoldGreedily(
            funcOp, std::move(threadLevelTilingPatterns));
        applyCanonicalizationPatternsForTiling(context, funcOp);
      }
      {
        OwningRewritePatternList patterns(context);
        // Apply canonicalization patterns.
        linalg::populateLinalgTilingCanonicalizationPatterns(patterns);
        populateAffineMinSCFCanonicalizationPattern(patterns);
        (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
      }
    }
  }
};
}  // namespace

std::unique_ptr<OperationPass<IREE::HAL::ExecutableTargetOp>>
createTileAndDistributeToThreads() {
  return std::make_unique<TileAndDistributeToThreads>();
}

static PassRegistration<TileAndDistributeToThreads> pass(
    "iree-codegen-cuda-tile-and-distribute",
    "Pass to tile and distribute linalg ops within a workgroup.");

}  // namespace iree_compiler
}  // namespace mlir
