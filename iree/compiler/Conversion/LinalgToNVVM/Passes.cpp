// Copyright 2020 Google LLC
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

//===- Passes.cpp - Pipeline from HLO to Linalg to SPIR-V -----------------===//
//
// Implementation of conversion from XLA-HLO to Linalg to SPIR-V dialect.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Conversion/LinalgToNVVM/Passes.h"

#include "iree/compiler/Conversion/CodegenUtils/ForOpCanonicalization.h"
#include "iree/compiler/Conversion/Common/Passes.h"
#include "iree/compiler/Conversion/HLOToHLO/Passes.h"
#include "iree/compiler/Conversion/HLOToLinalg/Passes.h"
#include "iree/compiler/Dialect/Shape/Transforms/Passes.h"
#include "iree/compiler/Conversion/CodegenUtils/FunctionUtils.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/SCFToGPU/SCFToGPUPass.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/PassManager.h"
#include "iree/compiler/Dialect/IREE/IR/IREEDialect.h"
#include "iree/compiler/Dialect/IREE/IR/IREEOps.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"

// Hack to get convertToGPU pass.
#include "iree/compiler/Conversion/LinalgToSPIRV/Passes.h"

namespace mlir {
namespace iree_compiler {

namespace {


class ConvertFunc : public ConvertToLLVMPattern {
 public:
  explicit ConvertFunc(MLIRContext *context, LLVMTypeConverter &converter)
      : ConvertToLLVMPattern(mlir::FuncOp::getOperationName(), context,
                             converter, 100) {}

  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto funcOp = cast<FuncOp>(op);
    FunctionType fnType = funcOp.getType();
    (void)fnType;
    if (!funcOp.isPublic()) return failure();

    // illegal FuncOp must have 0 inputs.
    assert(fnType.getNumInputs() == 0 && fnType.getNumResults() == 0);

    // func foo(%packed_buffer_args: !llvm.ptr<!llvm.ptr<i8>>,
    //          %push_constant: !llvm.ptr<i32>,
    //          workgroup_id[3]: !llvm.ptr<!llvm.array<i32, 3>>,
    //          workgroup_count[3]: !llvm.ptr<!llvm.array<i32, 3>>,
    //          workgroup_size[3]: !llvm.ptr<!llvm.array<i32, 3>>)
    TypeConverter::SignatureConversion signatureConverter(/*numOrigInputs=*/0);
    MLIRContext *context = rewriter.getContext();
    SmallVector<Type, 8> llvmInputTypes;
    funcOp.walk([&](IREE::PlaceholderOp placeholderOp) {
      auto memrefType = placeholderOp.getType().cast<MemRefType>();
      Type elType = memrefType.getElementType();
      auto llvmType =
          LLVM::LLVMPointerType::get(elType, memrefType.getMemorySpace());
      llvmInputTypes.push_back(llvmType);
    });
    signatureConverter.addInputs(llvmInputTypes);

    // Construct newFunc with all attributes except return type & symbol name.
    SmallVector<NamedAttribute, 4> funcAttrs;
    for (auto attr : funcOp.getAttrs()) {
      if (attr.first == SymbolTable::getSymbolAttrName() ||
          attr.first == mlir::impl::getTypeAttrName()) {
        continue;
      }
      funcAttrs.push_back(attr);
    }

    auto llvmFuncType = LLVM::LLVMFunctionType::get(
        LLVM::LLVMVoidType::get(rewriter.getContext()), llvmInputTypes);
    auto newFuncOp = rewriter.create<LLVM::LLVMFuncOp>(
        funcOp.getLoc(), funcOp.getName(), llvmFuncType,
        LLVM::Linkage::External, funcAttrs);

    // Copy all of funcOp's operations into newFuncOp's body and perform region
    // type conversion.
    rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(),
                                newFuncOp.end());
    if (failed(rewriter.convertRegionTypes(&newFuncOp.getBody(), *typeConverter,
                                           &signatureConverter)))
      return failure();

    rewriter.eraseOp(funcOp);
    return success();
  }
};
class ConvertIREEPlaceholderOp : public ConvertToLLVMPattern {
 public:
  explicit ConvertIREEPlaceholderOp(MLIRContext *context,
                                    LLVMTypeConverter &converter)
      : ConvertToLLVMPattern(IREE::PlaceholderOp::getOperationName(), context,
                             converter) {}

  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    // Bail until nested under an LLVMFuncOp.
    auto llvmFuncOp = op->getParentOfType<LLVM::LLVMFuncOp>();
    if (!llvmFuncOp) return failure();
    assert(llvmFuncOp.getNumArguments() > 0);

    auto llvmTypeConverter = getTypeConverter();
    Location loc = op->getLoc();
    auto ireePlaceHolderOp = cast<IREE::PlaceholderOp>(op);
    IREE::HAL::InterfaceBindingSubspanOpAdaptor adaptor(operands);
    MemRefType memrefType =
        ireePlaceHolderOp.getResult().getType().dyn_cast<MemRefType>();
    auto elementType = typeConverter->convertType(memrefType.getElementType());

    // Fetch the interface binding op and extract the buffer index from void**.
    auto symbol = SymbolTable::lookupNearestSymbolFrom(
        op, op->getAttrOfType<SymbolRefAttr>("binding"));
    auto interfaceBindingOp = cast<IREE::HAL::InterfaceBindingOp>(symbol);
    Value llvmBufferBasePtr =
        llvmFuncOp.getArgument(interfaceBindingOp.binding());
    if (memrefType.hasStaticShape()) {
      auto desc = MemRefDescriptor::fromStaticShape(
          rewriter, loc, *getTypeConverter(), memrefType, llvmBufferBasePtr);
      rewriter.replaceOp(op, {desc});
    } else {
      assert(0 && "TODO: implement dynamic shape");
    }

    return success();
  }
};

/// A pass that replaces all occurrences of GPU device operations with their
/// corresponding NVVM equivalent.
///
/// This pass only handles device code and is not meant to be run on GPU host
/// code.
struct LowerGpuOpsToNVVMOpsPass_IREE
    : public PassWrapper<LowerGpuOpsToNVVMOpsPass_IREE,
                         OperationPass<ModuleOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect, NVVM::NVVMDialect>();
  }
  void runOnOperation() override {
    ModuleOp m = getOperation();

    /// Customize the bitwidth used for the device side index computations.
    LowerToLLVMOptions options = {/*useBarePtrCallConv =*/false,
                                  /*emitCWrappers =*/false,
                                  /*indexBitwidth =*/64,
                                  /*useAlignedAlloc =*/false};
    LLVMTypeConverter converter(m.getContext(), options);
    // Apply in-dialect lowering first. In-dialect lowering will replace ops
    // which need to be lowered further, which is not supported by a single
    // conversion pass.
    {
      OwningRewritePatternList patterns;
      populateGpuRewritePatterns(m.getContext(), patterns);
      (void)applyPatternsAndFoldGreedily(m, std::move(patterns));
    }
    {
      OwningRewritePatternList llvmPatterns;
      llvmPatterns.insert<ConvertFunc, ConvertIREEPlaceholderOp>(m.getContext(),
                                                                 converter);
      populateStdToLLVMConversionPatterns(converter, llvmPatterns);
      populateGpuToNVVMConversionPatterns(converter, llvmPatterns);
      LLVMConversionTarget target(getContext());
      populateStdToLLVMFuncOpConversionPattern(converter, llvmPatterns);
      configureGpuToNVVMConversionLegality(target);
      target.addDynamicallyLegalOp<FuncOp>([&](FuncOp funcOp) {
        if (isEntryPoint(funcOp)) return false;
        return true;
      });
      if (failed(applyPartialConversion(m, target, std::move(llvmPatterns))))
        signalPassFailure();
    }
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createLowerGpuOpsToNVVMOpsPass_IREE() {
  return std::make_unique<LowerGpuOpsToNVVMOpsPass_IREE>();
}

} // anonymous namespace

static PassRegistration<LowerGpuOpsToNVVMOpsPass_IREE> pass(
    "iree-codegen-convert-to-nvvm",
    "Perform final conversion from builtin/GPU/HAL/standard dialect to LLVM "
    "and NVVM dialects");

static void addLinalgToNVVMPasses(OpPassManager &pm) {
  //===--------------------------------------------------------------------===//
  // Initial clean up.
  //===--------------------------------------------------------------------===//
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  //pm.addPass(createLinalgTileAndFusePass(options));
  pm.addPass(createCanonicalizerPass());

  //===--------------------------------------------------------------------===//
  // Map to GPU processor IDs.
  //
  // Post-conditions:
  //   - loop.parallel ops are converted to loop.for ops and mapped to
  //     workgroups.
  //   - Linalg ops are converted to loop.for ops and mapped to workitems.
  //===--------------------------------------------------------------------===//
  pm.addPass(createConvertToGPUPass());
  pm.addPass(createLowerAffinePass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // SCF -> STD
  pm.addNestedPass<FuncOp>(createLowerToCFGPass());
  pm.addNestedPass<FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<FuncOp>(createCSEPass());

  // TODO: convert to NVVM.
  pm.addPass(createLowerGpuOpsToNVVMOpsPass_IREE());
}

void buildNVVMTransformPassPipeline(OpPassManager &pm) {
  pm.addPass(createDeclareNumWorkgroupsFnPass());
  pm.addPass(createInlinerPass());

  //===--------------------------------------------------------------------===//
  // Inject shape calculation for output buffers.
  //
  // Pre-conditions:
  //   - All transformations altering the tensor-level shapes have been done.
  //   - "Root" dynamic tensors all pass through a single shapex.tie_shape
  //     use which associates them to their shape.
  //   - Loose, non-associated shapex.get_ranked_shape ops can exist anywhere
  //     and will be resolved.
  // Post-conditions:
  //   - All dynamic tensors bridge through a shapex.tie_shape op with the
  //     appropriate shape.
  //   - No shapex.get_ranked_shape ops exist.
  //   - Shape folding and canonicalization has been done.
  //===--------------------------------------------------------------------===//
  pm.addNestedPass<FuncOp>(Shape::createTieDynamicShapesPass());
  pm.addNestedPass<FuncOp>(Shape::createMaterializeShapeCalculationsPass());
  pm.addNestedPass<FuncOp>(Shape::createHoistShapeCalculationsPass());

  //===--------------------------------------------------------------------===//
  // Convert XLA HLO ops to Linalg ops with buffer semantics.
  //
  // Post-conditions:
  //   - All XLA HLO ops are converted.
  //   - All Linalg ops are operating on buffers.
  //===--------------------------------------------------------------------===//
  pm.addNestedPass<FuncOp>(createDecomposeHLOClampPass());
  addHLOToLinalgOnBuffersPasses(pm);

  //===--------------------------------------------------------------------===//
  // Convert Linalg ops to SPIR-V ops.
  //
  // Post-conditions:
  //   - All Linalg/Loops/GPU/Affine/Standard ops are converted away.
  //   - The module contains the final spv.module ready for serialization.
  //===--------------------------------------------------------------------===//
  addLinalgToNVVMPasses(pm);
}

static PassPipelineRegistration<> linalgToNVVMPipeline(
    "iree-codegen-linalg-to-nvvm-pipeline",
    "Runs the progressive lowering pipeline from Linalg to NVVM",
    [](OpPassManager &passManager) {
      addLinalgToNVVMPasses(passManager);
    });

static PassPipelineRegistration<> hloToLinalgNVVMPipeline(
    "iree-codegen-hlo-to-nvvm-pipeline",
    "Runs the progressive lowering pipeline from XLA HLO to Linalg to "
    "NVVM",
    [](OpPassManager &passManager) {
      buildNVVMTransformPassPipeline(passManager);
    });

}  // namespace iree_compiler
}  // namespace mlir
