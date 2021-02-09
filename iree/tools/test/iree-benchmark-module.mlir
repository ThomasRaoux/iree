// RUN: iree-translate --iree-hal-target-backends=vmla -iree-mlir-to-vm-bytecode-module %s | iree-benchmark-module --driver=vmla --entry_function=abs --function_inputs="i32=-2" | IreeFileCheck %s
// RUN: [[ $IREE_VULKAN_DISABLE == 1 ]] || (iree-translate --iree-hal-target-backends=vulkan-spirv -iree-mlir-to-vm-bytecode-module %s | iree-benchmark-module --driver=vulkan --entry_function=abs --function_inputs="i32=-2" | IreeFileCheck %s)
// RUN: [[ $IREE_LLVMAOT_DISABLE == 1 ]] || (iree-translate --iree-hal-target-backends=dylib-llvm-aot -iree-mlir-to-vm-bytecode-module %s | iree-benchmark-module --driver=dylib --entry_function=abs --function_inputs="i32=-2" | IreeFileCheck %s)

// CHECK-LABEL: BM_abs
func @abs(%input : tensor<16xf32>, %input1 : tensor<16xf32>) -> (tensor<16xf32>) attributes { iree.module.export } {
  %result = "mhlo.add"(%input, %input1) : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
  return %result : tensor<16xf32>
}
