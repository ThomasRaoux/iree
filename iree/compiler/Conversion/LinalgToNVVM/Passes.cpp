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

#include "iree/compiler/Conversion/LinalgToNVVM/Passes.h"

#include "iree/compiler/Conversion/Common/Passes.h"
#include "iree/compiler/Conversion/HLOToHLO/Passes.h"
#include "iree/compiler/Conversion/HLOToLinalg/Passes.h"
#include "iree/compiler/Dialect/Shape/Transforms/Passes.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace iree_compiler {

static void addLinalgToNVVMPasses(OpPassManager &pm) {
  //===--------------------------------------------------------------------===//
  // Initial clean up.
  //===--------------------------------------------------------------------===//
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // TODO: This currently maps to a single thread. We should share Tile and
  // distribute with other GPU backends.
  // Linalg -> SCF
  pm.addNestedPass<FuncOp>(createConvertLinalgToLoopsPass());
  pm.addNestedPass<FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<FuncOp>(createCSEPass());

  // SCF -> STD
  pm.addNestedPass<FuncOp>(createLowerToCFGPass());
  pm.addNestedPass<FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<FuncOp>(createCSEPass());

  // Strip out the debug info for the kernel as CUDA driver doesn't diggest PTX
  // debug info well.
  pm.addPass(createStripDebugInfoPass());
  // convert to NVVM.
  pm.addPass(createConvertToNVVMPass());
}

void buildNVVMTransformPassPipeline(OpPassManager &pm) {
  OpPassManager &nestedModulePM = pm.nest<ModuleOp>();
  nestedModulePM.addPass(createDeclareNumWorkgroupsFnPass());
  nestedModulePM.addPass(createInlinerPass());

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
  nestedModulePM.addNestedPass<FuncOp>(Shape::createTieDynamicShapesPass());
  nestedModulePM.addNestedPass<FuncOp>(
      Shape::createMaterializeShapeCalculationsPass());
  nestedModulePM.addNestedPass<FuncOp>(
      Shape::createHoistShapeCalculationsPass());

  //===--------------------------------------------------------------------===//
  // Convert XLA HLO ops to Linalg ops with buffer semantics.
  //
  // Post-conditions:
  //   - All XLA HLO ops are converted.
  //   - All Linalg ops are operating on buffers.
  //===--------------------------------------------------------------------===//
  nestedModulePM.addNestedPass<FuncOp>(createDecomposeHLOClampPass());
  addHLOToLinalgOnBuffersPasses(nestedModulePM);

  //===--------------------------------------------------------------------===//
  // Convert Linalg ops to SPIR-V ops.
  //
  // Post-conditions:
  //   - All Linalg/Loops/GPU/Affine/Standard ops are converted away.
  //   - The module contains the final spv.module ready for serialization.
  //===--------------------------------------------------------------------===//
  addLinalgToNVVMPasses(nestedModulePM);
}

static PassPipelineRegistration<> linalgToNVVMPipeline(
    "iree-codegen-linalg-to-nvvm-pipeline",
    "Runs the progressive lowering pipeline from Linalg to NVVM",
    [](OpPassManager &passManager) { addLinalgToNVVMPasses(passManager); });

static PassPipelineRegistration<> hloToLinalgNVVMPipeline(
    "iree-codegen-hlo-to-nvvm-pipeline",
    "Runs the progressive lowering pipeline from XLA HLO to Linalg to "
    "NVVM",
    [](OpPassManager &passManager) {
      buildNVVMTransformPassPipeline(passManager);
    });

}  // namespace iree_compiler
}  // namespace mlir
