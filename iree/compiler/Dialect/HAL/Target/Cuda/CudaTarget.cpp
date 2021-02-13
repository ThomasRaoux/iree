// Copyright 2019 Google LLC
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

#include "iree/compiler/Dialect/HAL/Target/Cuda/CudaTarget.h"

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/Utils/FlatbufferUtils.h"
#include "iree/schemas/cuda_executable_def_builder.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "iree/compiler/Conversion/LinalgToNVVM/Passes.h"
#include "mlir/Target/LLVMIR.h"
#include "mlir/Target/NVVMIR.h"
#include "llvm/Support/TargetSelect.h"

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Mutex.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

CudaTargetOptions getCudaTargetOptionsFromFlags() {
  CudaTargetOptions targetOptions;
  // TODO: flags
  return targetOptions;
}

static std::string translateModuleToISA(
    llvm::Module &module, llvm::TargetMachine &targetMachine) {
  std::string targetISA;
  {
    llvm::raw_string_ostream stream(targetISA);
    llvm::buffer_ostream pstream(stream);
    llvm::legacy::PassManager codegenPasses;
    targetMachine.addPassesToEmitFile(codegenPasses, pstream, nullptr,
                                      llvm::CGFT_AssemblyFile);
    codegenPasses.run(module);
  }
  return targetISA;
}

class CudaTargetBackend final : public TargetBackend {
 public:
  CudaTargetBackend(CudaTargetOptions options) : options_(std::move(options)) {}

  std::string name() const override { return "cuda"; }
  std::string filter_pattern() const override { return "cuda"; }

  void buildTranslationPassPipeline(OpPassManager &passManager) override {
    OpPassManager &nestedModulePM = passManager.nest<ModuleOp>();
    buildNVVMTransformPassPipeline(nestedModulePM);
  }

  LogicalResult serializeExecutable(IREE::HAL::ExecutableTargetOp targetOp,
                                    OpBuilder &executableBuilder) override {
    // Perform the translation in a separate context to avoid any
    // multi-threading issues.
    llvm::LLVMContext context;

    // We name our files after the executable name so that they are easy to
    // track both during compilation (logs/artifacts/etc), as outputs (final
    // intermediate code/binary files), and at runtime (loaded
    // libraries/symbols/etc).
    auto libraryName =
        targetOp->getParentOfType<IREE::HAL::ExecutableOp>().getName().str();

    ModuleOp innerModuleOp = targetOp.getInnerModule();

    // TODO(#3737): don't add functions we don't want to serialize to the
    // module. Right now workgroup count calculation functions end up in here
    // as std.func ops and not just the llvm.func ops we expect.
    auto illegalFuncOps = llvm::to_vector<4>(innerModuleOp.getOps<FuncOp>());
    for (auto funcOp : illegalFuncOps) {
      funcOp.erase();
    }
    auto halInterfaceOps =
        llvm::to_vector<1>(innerModuleOp.getOps<IREE::HAL::InterfaceOp>());
    for (auto halOp : halInterfaceOps) {
      halOp.erase();
    }
    // Workaround: Invalidate the debug location on purpose as CUDA driver
    // doesn't seem to diggest the debug info well.
    innerModuleOp.walk([&](Operation *op) {
      op->setLoc(UnknownLoc::get(innerModuleOp.getContext()));
    });

    auto llvmModule =
        mlir::translateModuleToNVVMIR(innerModuleOp, context, libraryName);
    if (!llvmModule) {
      return targetOp.emitError() << "failed to translate the MLIR LLVM "
                                     "dialect to the native llvm::Module";
    }
    for (auto func : innerModuleOp.getOps<LLVM::LLVMFuncOp>()) {
      auto *llvmFunc = llvmModule->getFunction(func.getName());

      llvm::Metadata *llvmMetadata[] = {
          llvm::ValueAsMetadata::get(llvmFunc),
          llvm::MDString::get(llvmModule->getContext(), "kernel"),
          llvm::ValueAsMetadata::get(llvm::ConstantInt::get(
              llvm::Type::getInt32Ty(llvmModule->getContext()), 1))};
      llvm::MDNode *llvmMetadataNode =
          llvm::MDNode::get(llvmModule->getContext(), llvmMetadata);
      llvmModule->getOrInsertNamedMetadata("nvvm.annotations")
          ->addOperand(llvmMetadataNode);
    }

    std::unique_ptr<llvm::TargetMachine> targetMachine;
    {
      llvm::Triple triple("nvptx64-nvidia-cuda");
      std::string targetChip = "sm_35";
      std::string features = "+ptx60";
      std::string error;
      const llvm::Target *target =
          llvm::TargetRegistry::lookupTarget("", triple, error);
      if (target == nullptr) {
        return targetOp.emitError() << "cannot initialize target triple";
      }
      targetMachine.reset(target->createTargetMachine(triple.str(), targetChip,
                                                      features, {}, {}));
      if (targetMachine == nullptr) {
        return targetOp.emitError() << "cannot initialize target machine";
      }
    }

    llvmModule->setDataLayout(targetMachine->createDataLayout());

    std::string targetISA = translateModuleToISA(*llvmModule, *targetMachine);
    // Serialize cuda kernel into the binary that we will embed in the
    // final flatbuffer.
    FlatbufferBuilder builder;
    auto ptxCudeRef = flatbuffers_uint8_vec_create(
        builder, reinterpret_cast<const uint8_t *>(targetISA.c_str()),
        targetISA.size());

    auto entryPointNames = llvm::to_vector<8>(
        llvm::map_range(targetOp.getBlock().getOps<ExecutableEntryPointOp>(),
                        [&](auto op) { return op.getName(); }));
    auto entryPointsRef = builder.createStringVec(entryPointNames);

    // iree_CudaThreadgroupSize_vec_start(builder);
    // for (auto &shader : mslShaders) {
    //  iree_CudaThreadgroupSize_vec_push_create(
    //      builder, 16, 1, 1);
    // }
    // auto threadgroupSizesRef = iree_CudaThreadgroupSize_vec_end(builder);

    iree_CudaExecutableDef_start_as_root(builder);
    iree_CudaExecutableDef_entry_points_add(builder, entryPointsRef);
    // iree_CudaExecutableDef_threadgroup_sizes_add(builder,
    // threadgroupSizesRef);
    iree_CudaExecutableDef_kernel_library_add(builder, ptxCudeRef);
    iree_CudaExecutableDef_end_as_root(builder);

    // Add the binary data to the target executable.
    executableBuilder.create<IREE::HAL::ExecutableBinaryOp>(
        targetOp.getLoc(), targetOp.sym_name(),
        static_cast<uint32_t>(IREE::HAL::ExecutableFormat::Cuda),
        builder.getBufferAttr(executableBuilder.getContext()));

    return success();
  }

  std::array<Value, 3> calculateDispatchWorkgroupCount(
      Location loc, IREE::HAL::ExecutableOp executableOp,
      IREE::HAL::ExecutableEntryPointOp entryPointOp, ValueRange workload,
      OpBuilder &builder) override {
    // For now we are not tiling and just dispatch everything as 1,1,1.
    auto constantOne = builder.createOrFold<mlir::ConstantIndexOp>(loc, 1);
    return {constantOne, constantOne, constantOne};
  }

 private:
  CudaTargetOptions options_;
};

void registerCudaTargetBackends(
    std::function<CudaTargetOptions()> queryOptions) {
  getCudaTargetOptionsFromFlags();
  static TargetBackendRegistration registration("cuda", [=]() {
#define INIT_LLVM_TARGET(TargetName)        \
  LLVMInitialize##TargetName##Target();     \
  LLVMInitialize##TargetName##TargetMC();   \
  LLVMInitialize##TargetName##TargetInfo(); \
  LLVMInitialize##TargetName##AsmPrinter();
    INIT_LLVM_TARGET(NVPTX)
    return std::make_unique<CudaTargetBackend>(queryOptions());
  });
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
