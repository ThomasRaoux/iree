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

#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"

/// This file contains patterns allowing canonicalization taking advantage  of
/// flow dispatch ops. When the ID range is know we may be able to calculate the
/// min/max of the derived values and may allow folding affine.min or ForOp.
/// This file duplicate some of the logic from
/// `linalg::AffineMinSCFCanonicalizationPattern`. In the future we could unify
/// this code back by upstreaminng an interface to MLIR to represent generic ID
/// ops.
namespace mlir {
namespace iree_compiler {

namespace {

static void combineAndSimplifyMap(AffineMap &map, SmallVectorImpl<Value> &dims,
                   SmallVectorImpl<Value> &symbols) {
  SmallVector<Value, 4> operands(dims.begin(), dims.end());
  operands.append(symbols.begin(), symbols.end());
  // Pull in affine.apply operations and compose them fully into the
  // result.
  fullyComposeAffineMapAndOperands(&map, &operands);
  canonicalizeMapAndOperands(&map, &operands);
  map = simplifyAffineMap(map);
  // Assign the results.
  dims.assign(operands.begin(), operands.begin() + map.getNumDims());
  symbols.assign(operands.begin() + map.getNumDims(), operands.end());
}

/// Traverse the `dims` and substitute known min or max expressions in place of
/// values whose range is known. We know the range of some operations based on
/// their semantic.
static AffineMap substituteID(AffineMap map, SmallVectorImpl<Value> &dims,
                              SmallVectorImpl<Value> &symbols,
                              ArrayRef<int32_t> workgroupSize) {
  // First pull in affine.apply and compose them.
  combineAndSimplifyMap(map, dims, symbols);
  auto exprs = llvm::to_vector<4>(map.getResults());
  for (AffineExpr &expr : exprs) {
    bool substituted = true;
    while (substituted) {
      substituted = false;
      for (unsigned dimIdx = 0; dimIdx < dims.size(); ++dimIdx) {
        Value dim = dims[dimIdx];
        AffineExpr dimExpr = getAffineDimExpr(dimIdx, expr.getContext());
        AffineExpr substitutedExpr;
        if (auto idOp = dim.getDefiningOp<gpu::ThreadIdOp>()) {
          unsigned index = StringSwitch<unsigned>(idOp.dimension())
                               .Case("x", 0)
                               .Case("y", 1)
                               .Case("z", 2);
          OpBuilder b(map.getContext());
          AffineExpr zero = b.getAffineConstantExpr(0);
          AffineExpr ubExpr = b.getAffineConstantExpr(workgroupSize[index]);
          substitutedExpr = substWithMin(expr, dimExpr, zero, ubExpr - 1);
        }
        if (!substitutedExpr) continue;
        substituted = (substitutedExpr != expr);
        expr = substitutedExpr;
      }
      for (unsigned symIdx = 0; symIdx < symbols.size(); ++symIdx) {
        Value sym = symbols[symIdx];
        AffineExpr symExpr = getAffineSymbolExpr(symIdx, expr.getContext());
        AffineExpr substitutedExpr;
        if (auto idOp = sym.getDefiningOp<gpu::ThreadIdOp>()) {
          unsigned index = StringSwitch<unsigned>(idOp.dimension())
                               .Case("x", 0)
                               .Case("y", 1)
                               .Case("z", 2);
          OpBuilder b(map.getContext());
          AffineExpr zero = b.getAffineConstantExpr(0);
          AffineExpr ubExpr = b.getAffineConstantExpr(workgroupSize[index]);
          substitutedExpr = substWithMin(expr, symExpr, zero, ubExpr - 1);
        }
        if (!substitutedExpr) continue;
        substituted = (substitutedExpr != expr);
        expr = substitutedExpr;
      }
    }

    // Cleanup and simplify the results.
    // This needs to happen outside of the loop iterating on dims.size() since
    // it modifies dims.
    auto map = AffineMap::get(dims.size(), symbols.size(), exprs,
                              exprs.front().getContext());
    combineAndSimplifyMap(map, dims, symbols);
    // Assign the results.
    exprs.assign(map.getResults().begin(), map.getResults().end());
  }

  assert(!exprs.empty() && "Unexpected empty exprs");
  return AffineMap::get(dims.size(), symbols.size(), exprs, map.getContext());
}

/// Replaces the given op with the contents of the given single-block region,
/// using the operands of the block terminator to replace operation results.
static void replaceOpWithRegion(PatternRewriter &rewriter, Operation *op,
                                Region &region, ValueRange blockArgs = {}) {
  assert(llvm::hasSingleElement(region) && "expected single-region block");
  Block *block = &region.front();
  Operation *terminator = block->getTerminator();
  ValueRange results = terminator->getOperands();
  rewriter.mergeBlockBefore(block, op, blockArgs);
  rewriter.replaceOp(op, results);
  rewriter.eraseOp(terminator);
}

/// Return true if we can prove that the we always run at least the first
/// iteration of the ForOp.
static bool alwaysRunsFirstIteration(scf::ForOp op,
                                     ArrayRef<int32_t> workgroupSize) {
  // Calculate the minimum value of ub - lb. If it is strictly positive it
  // means the loop will always run at least once.
  MLIRContext *ctx = op->getContext();
  SmallVector<Value, 4> dims;
  SmallVector<Value, 4> symbols;
  AffineExpr lb = getAffineDimExpr(dims.size(), ctx);
  dims.push_back(op.lowerBound());
  AffineExpr ub = getAffineDimExpr(dims.size(), ctx);
  dims.push_back(op.upperBound());
  AffineExpr iterZero = ub - lb;
  auto map = AffineMap::get(dims.size(), 0, iterZero);
  AffineMap simplifiedMap = substituteID(map, dims, symbols, workgroupSize);
  assert(simplifiedMap.getNumResults() == 1);
  if (auto cst = simplifiedMap.getResult(0).dyn_cast<AffineConstantExpr>()) {
    if (cst.getValue() > 0) return true;
  }
  return false;
}

/// Return true if we can prove that the we never run more than one iteration of
/// the ForOp.
static bool neverRunsSecondIteration(scf::ForOp op,
                                     ArrayRef<int32_t> workgroupSize) {
  // Calculate the minimum of lb + step - ub. If it is positive it means the
  // loop never run more than once.
  MLIRContext *ctx = op->getContext();
  SmallVector<Value, 4> dims;
  SmallVector<Value, 4> symbols;
  AffineExpr lb = getAffineDimExpr(dims.size(), ctx);
  dims.push_back(op.lowerBound());
  AffineExpr ub = getAffineDimExpr(dims.size(), ctx);
  dims.push_back(op.upperBound());
  AffineExpr step = getAffineDimExpr(dims.size(), ctx);
  dims.push_back(op.step());
  AffineExpr iterOne = lb + step - ub;
  auto map = AffineMap::get(dims.size(), 0, iterOne);

  AffineMap simplifiedMap = substituteID(map, dims, symbols, workgroupSize);
  assert(simplifiedMap.getNumResults() == 1);
  if (auto cst = simplifiedMap.getResult(0).dyn_cast<AffineConstantExpr>()) {
    if (cst.getValue() >= 0) return true;
  }
  return false;
}

/// Rewriting pattern that replaces single-iteration loops with their bodies.
struct SimplifyTrivialLoops : public OpRewritePattern<scf::ForOp> {
  SimplifyTrivialLoops(
      MLIRContext *context, std::array<int32_t, 3> &workgroupSize)
      : OpRewritePattern<scf::ForOp>(context, 1), workgroupSize(workgroupSize){}

  LogicalResult matchAndRewrite(scf::ForOp op,
                                PatternRewriter &rewriter) const override {
    // TODO: Handle the case where we know that the loop doesn't run more than
    // once but the loop may not run at least once by replace the `loop` with an
    // `if`.
    if (!(alwaysRunsFirstIteration(op, workgroupSize) &&
          neverRunsSecondIteration(op, workgroupSize)))
      return failure();

    // The first iteration is always run and the second iteration is never run
    // so the loop always have 1 iteration. Inline its body and remove the loop.
    SmallVector<Value, 4> blockArgs;
    blockArgs.reserve(op.getNumIterOperands() + 1);
    blockArgs.push_back(op.lowerBound());
    llvm::append_range(blockArgs, op.getIterOperands());
    replaceOpWithRegion(rewriter, op, op.getLoopBody(), blockArgs);
    return success();
  }
  private:
  std::array<int32_t, 3> workgroupSize;
};

class SimplifyTrivialLoopPass
    : public PassWrapper<SimplifyTrivialLoopPass,
                         OperationPass<FuncOp>> {
  void runOnOperation() override {
    FuncOp funcOp = getOperation();
    std::array<int32_t, 3> workgroupSize;
    for (auto it : llvm::enumerate(funcOp->getAttr("cuda_workgroup_size")
                                       .cast<DenseIntElementsAttr>()
                                       .getIntValues())) {
      workgroupSize[it.index()] = it.value().getZExtValue();
    }
    MLIRContext *context = funcOp->getContext();
    OwningRewritePatternList patterns(context);
    patterns.insert<SimplifyTrivialLoops>(
        context, workgroupSize);
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
  }
};

}  // namespace


std::unique_ptr<OperationPass<FuncOp>> createSimplifyTrivialLoopPass() {
  return std::make_unique<SimplifyTrivialLoopPass>();
}

static PassRegistration<SimplifyTrivialLoopPass> pass(
    "iree-flow-dispatch-id-canonicalizations",
    "Canonicalization patterns related to flow disptach ops.");

}  // namespace iree_compiler
}  // namespace mlir