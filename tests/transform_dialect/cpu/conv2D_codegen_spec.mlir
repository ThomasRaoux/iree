// RUN: iree-opt %s

// ./build/tools/iree-opt ./tests/transform_dialect/cpu/conv2D.mlir --iree-hal-target-backends=llvm-cpu   --iree-abi-transformation-pipeline   --iree-flow-transformation-pipeline    --iree-flow-dispatch-use-transform-dialect=tests/transform_dialect/cpu/conv2D_dispatch_spec.mlir --iree-stream-transformation-pipeline   --iree-hal-configuration-pipeline | ./build/tools/iree-opt --pass-pipeline='hal.executable(hal.executable.variant(iree-llvmcpu-lower-executable-target))' --iree-codegen-llvmcpu-use-transform-dialect=tests/transform_dialect/cpu/conv2D_codegen_spec.mlir

// ./build/tools/iree-compile ./tests/transform_dialect/cpu/conv2D.mlir   --iree-hal-target-backends=llvm-cpu   --iree-flow-dispatch-use-transform-dialect=tests/transform_dialect/cpu/conv2D_dispatch_spec.mlir   --iree-codegen-llvmcpu-use-transform-dialect=./tests/transform_dialect/cpu/conv2D_codegen_spec.mlir   --iree-llvm-target-triple=x86_64-pc-linux-gnu   --iree-llvm-target-cpu-features=host   --iree-hal-benchmark-dispatch-repeat-count=10000 |   ./build/tools/iree-benchmark-module --device=local-task --task_topology_group_count=16 --batch_size=10000  --entry_function=conv2d_1x230x230x3_7x7x3x64 

transform.structured.canonicalized_sequence failures(propagate) {
^bb1(%variant_op: !pdl.operation):
  %fill = transform.structured.match ops{["linalg.fill"]} in %variant_op
  %conv_2d = transform.structured.match ops{["linalg.conv_2d_nhwc_hwcf"]} in %variant_op

  %foreach_thread, %tiled_parallel_conv_2d =                                  // n  h
    transform.structured.tile_to_foreach_thread_op %conv_2d tile_sizes [0, 1]
  
  // // TODO: Enable this to fuse the fill.
  // transform.structured.fuse_into_containing_op %fill into %foreach_thread
  // %tiled_conv_2d, %loops:5 = transform.structured.fuse %tiled_parallel_conv_2d 
  // //               n  h  w   c  kh  kw
  //   {tile_sizes = [0, 1, 2, 16,  1,  1], tile_interchange = [0, 1, 2, 3, 4, 5]}


  %tiled_conv_2d, %loops:5 =                    // n  h  w  c  kh  kw
    transform.structured.tile %tiled_parallel_conv_2d [0, 1, 2, 16,  1,  1]

  %tiled_conv_1d = transform.structured.decompose %tiled_conv_2d
  
  // TODO: Generalize followed by rank-reduction drops all 1s and breaks the
  // batched conv_1d pattern and we miss vectorization.
  // %tiled_generic = transform.structured.generalize %tiled_conv
  
  %func = transform.structured.match ops{["func.func"]} in %variant_op
  %func_3 = transform.structured.vectorize %func

  transform.iree.apply_patterns %func_3 { rank_reducing }
  %variant_op_2 = transform.iree.bufferize %variant_op

  %func_4 = transform.structured.match ops{["func.func"]} in %variant_op_2
  transform.iree.foreach_thread_to_workgroup %func_4
}
