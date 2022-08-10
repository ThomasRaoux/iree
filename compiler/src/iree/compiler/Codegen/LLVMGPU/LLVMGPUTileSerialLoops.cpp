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
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"


namespace mlir {
namespace iree_compiler {

/// Patterns for workgroup level tiling. Workgroup tiling is done at the flow
/// level but we may have extra tiling for the reduction dimension. Therefore we
/// tile again without distributing.
static void populateTilingReductionPatterns(RewritePatternSet &patterns) {
  auto tileSizesFn = [&](OpBuilder &builder,
                         Operation *op) -> SmallVector<Value, 4> {
    auto interfaceOp = cast<PartitionableLoopsInterface>(*op);
    unsigned numLoops = interfaceOp.getNumLoops();
    SmallVector<Value, 4> tileSizes;
    tileSizes = getTileSizes(builder, op, 1);
    if(tileSizes.empty())
      return tileSizes;
    if(numLoops > tileSizes.size()) {
      auto one = builder.create<arith::ConstantIndexOp>(op->getLoc(), 1);
      auto zero = builder.create<arith::ConstantIndexOp>(op->getLoc(), 0);
      SmallVector<Value, 4> tileSizesTemp(numLoops - 2, zero);
      tileSizesTemp.push_back(one);
      tileSizesTemp.push_back(tileSizes.back());
      tileSizes = tileSizesTemp;
    }
    tileSizes.resize(numLoops);
    return tileSizes;
  };

  auto tilingOptions = linalg::LinalgTilingOptions()
                           .setLoopType(linalg::LinalgTilingLoopType::Loops)
                           .setTileSizeComputationFunction(tileSizesFn);
  MLIRContext *context = patterns.getContext();

  linalg::LinalgTransformationFilter filter(
      ArrayRef<StringAttr>{StringAttr::get(patterns.getContext(), "SPLIT")},
      StringAttr::get(context, getVectorizeMarker()));
  filter.setMatchByDefault();
  linalg::TilingPatterns<linalg::MatmulOp, linalg::BatchMatmulOp,
                         linalg::GenericOp>::insert(patterns, tilingOptions,
                                                    filter);
}

static std::pair<int64_t, unsigned> splitReductionControl(linalg::LinalgOp op) {
  return std::make_pair(1024, 3);
}

namespace {
struct LinalgSplitReduction
    : public OpInterfaceRewritePattern<linalg::LinalgOp> {
  LinalgSplitReduction(MLIRContext *context,
                       linalg::ControlSplitReductionFn controlSplitReductionFn,
                       linalg::LinalgTransformationFilter f,
                       PatternBenefit benefit = 1)
      : OpInterfaceRewritePattern<linalg::LinalgOp>(context, benefit),
        controlSplitReductionFn(controlSplitReductionFn),
        filter(std::move(f)) {}

  LogicalResult matchAndRewrite(linalg::LinalgOp op,
                                PatternRewriter &rewriter) const override {
    std::vector<std::pair<StringAttr, Attribute>> attributes;
    // Since user information about compilation are passed through attributes we
    // need to make sure to propagate those.
    if (auto generic = dyn_cast<linalg::GenericOp>(op.getOperation())) {
      ArrayRef<StringRef> odsAttrs = generic.getAttributeNames();
      for (NamedAttribute kv : op->getAttrs()) {
        if (!llvm::is_contained(odsAttrs, kv.getName().getValue())) {
          attributes.push_back(std::make_pair(kv.getName(), kv.getValue()));
        }
      }
    }

    FailureOr<linalg::LinalgOp> result =
        splitReduction(rewriter, op, controlSplitReductionFn, filter);
    if (failed(result)) return failure();
    // If any attributes needs to be propagated set it.
    for (std::pair<StringAttr, Attribute> &attrib : attributes) {
      result.getValue()->setAttr(attrib.first, attrib.second);
    }
    return result;
  }

 private:
  linalg::ControlSplitReductionFn controlSplitReductionFn;
  linalg::LinalgTransformationFilter filter;
};


struct LLVMGPUTileSerialLoopsPass
    : public LLVMGPUTileSerialLoopsBase<LLVMGPUTileSerialLoopsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect>();
  }
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto funcOp = getOperation();

    {
      RewritePatternSet splitReductionPatterns(context);
      linalg::LinalgTransformationFilter filter(
          ArrayRef<StringAttr>{}, StringAttr::get(&getContext(), "SPLIT"));
      filter.setMatchByDefault();
      splitReductionPatterns.add<LinalgSplitReduction>(
          context, splitReductionControl, filter);
      if (failed(applyPatternsAndFoldGreedily(
              funcOp, std::move(splitReductionPatterns)))) {
        return signalPassFailure();
      }
    }

    // Tile again at the workgroup level since redution dimension were
    // ignored. Dimensions already tiled will be ignore since we tile to the
    // same size.
    RewritePatternSet wgTilingPatterns(context);
    populateTilingReductionPatterns(wgTilingPatterns);
    if (failed(applyPatternsAndFoldGreedily(funcOp,
                                            std::move(wgTilingPatterns)))) {
      return signalPassFailure();
    }
  }
};
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createLLVMGPUTileSerialLoops() {
  return std::make_unique<LLVMGPUTileSerialLoopsPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
