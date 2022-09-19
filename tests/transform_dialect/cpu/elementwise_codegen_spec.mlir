// RUN: iree-opt %s

// ./build/tools/iree-opt ./tests/transform_dialect/cpu/elementwise.mlir   --iree-hal-target-backends=llvm-cpu   --iree-abi-transformation-pipeline   --iree-flow-transformation-pipeline    --iree-flow-dispatch-use-transform-dialect=tests/transform_dialect/cpu/elementwise_dispatch_spec.mlir    --iree-stream-transformation-pipeline   --iree-hal-configuration-pipeline | ./build/tools/iree-opt --pass-pipeline='hal.executable(hal.executable.variant(iree-llvmcpu-lower-executable-target))'  --iree-codegen-llvmcpu-use-transform-dialect=tests/transform_dialect/cpu/elementwise_codegen_spec.mlir | head -n -26
// ./build/tools/iree-compile ./tests/transform_dialect/cpu/elementwise.mlir   --iree-hal-target-backends=llvm-cpu   --iree-flow-dispatch-use-transform-dialect=tests/transform_dialect/cpu/elementwise_dispatch_spec.mlir   --iree-codegen-llvmcpu-use-transform-dialect=./tests/transform_dialect/cpu/elementwise_codegen_spec.mlir   --iree-llvm-target-triple=x86_64-pc-linux-gnu   --iree-llvm-target-cpu-features=host   --iree-hal-benchmark-dispatch-repeat-count=1000 |   ./build/tools/iree-benchmark-module --device=local-task --task_topology_group_count=0 --batch_size=1000     --entry_function=elementwise --function_input="16xf32=1" --function_input="16xf32=2" --function_input="16xf32=3"

transform.structured.canonicalized_sequence failures(propagate) {
^bb1(%variant_op: !pdl.operation):

  %generic = transform.structured.match ops{["linalg.generic"]} in %variant_op
  
  %foreach_thread, %tiled_generic =
    transform.structured.tile_to_foreach_thread_op %generic num_threads [2]

  %func = transform.structured.match ops{["func.func"]} in %variant_op  
  %func_2 = transform.structured.vectorize %func

  %variant_op_2 = transform.iree.bufferize %variant_op
  %func_3 = transform.structured.match ops{["func.func"]} in %variant_op_2  
  %func_4 = transform.iree.foreach_thread_to_workgroup %func_3
}
