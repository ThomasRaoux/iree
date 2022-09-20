// RUN: iree-opt %s

// ./build/tools/iree-opt ./tests/transform_dialect/cpu/reduction.mlir --iree-hal-target-backends=llvm-cpu   --iree-abi-transformation-pipeline   --iree-flow-transformation-pipeline    --iree-flow-dispatch-use-transform-dialect=tests/transform_dialect/cpu/reduction_dispatch_spec.mlir --iree-stream-transformation-pipeline   --iree-hal-configuration-pipeline | ./build/tools/iree-opt --pass-pipeline='hal.executable(hal.executable.variant(iree-llvmcpu-lower-executable-target))' --iree-codegen-llvmcpu-use-transform-dialect=tests/transform_dialect/cpu/reduction_codegen_spec.mlir | head -n -26
// ./build/tools/iree-compile ./tests/transform_dialect/cpu/reduction.mlir   --iree-hal-target-backends=llvm-cpu   --iree-flow-dispatch-use-transform-dialect=tests/transform_dialect/cpu/reduction_dispatch_spec.mlir   --iree-codegen-llvmcpu-use-transform-dialect=./tests/transform_dialect/cpu/reduction_codegen_spec.mlir   --iree-llvm-target-triple=x86_64-pc-linux-gnu   --iree-llvm-target-cpu-features=host   --iree-hal-benchmark-dispatch-repeat-count=10000 |   ./build/tools/iree-benchmark-module --device=local-task --task_topology_group_count=16 --batch_size=10000     --entry_function=reduce 
transform.structured.canonicalized_sequence failures(propagate) {
^bb1(%variant_op: !pdl.operation):
  %fill = transform.structured.match ops{["linalg.fill"]} in %variant_op

  // Split the reduction by 2 to obtain a more meaty parallel op with
  // parallelism across size(reduction) / 2 threads.
  %generic = transform.structured.match ops{["linalg.generic"]} in %variant_op

  // First level of tiling + fusion parallelizes to blocks.
  // The mapping to block ids can only happen after bufferization atm.
  %foreach_thread_grid, %tiled_generic =
    transform.structured.tile_to_foreach_thread_op %generic tile_sizes [1]
  transform.structured.fuse_into_containing_op %fill into %foreach_thread_grid

  transform.structured.tile %tiled_generic [0, 64]
  %func = transform.structured.match ops{["func.func"]} in %variant_op
  %func_2 = transform.iree.apply_patterns %func { rank_reducing }
  %func_3 = transform.structured.vectorize %func_2

  // TODO(springerm): there is an extra roundtrip copy here.
  %variant_op_2 = transform.iree.bufferize %variant_op
  %func_4 = transform.structured.match ops{["func.func"]} in %variant_op_2

  %func_5 = transform.iree.foreach_thread_to_workgroup %func_4

  lower_vectors { multireduction_lowering = "innerreduce"}
}
