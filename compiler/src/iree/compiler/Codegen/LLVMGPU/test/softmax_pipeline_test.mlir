// RUN: iree-opt --split-input-file --pass-pipeline='hal.executable(hal.executable.variant(iree-llvmgpu-lower-executable-target-pass))' %s | FileCheck %s

#device_target_cuda = #hal.device.target<"cuda", {executable_targets = [#hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_35"}>], legacy_sync}>
#executable_target_cuda_nvptx_fb = #hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_35"}>
#map0 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>
module attributes {hal.device.targets = [#device_target_cuda]} {
  hal.executable private @softmax_dispatch_0 {
    hal.executable.variant public @cuda_nvptx_fb, target = #executable_target_cuda_nvptx_fb {
      hal.executable.export public @softmax_dispatch_0_generic_12x128x128 ordinal(0) layout(#pipeline_layout) {
      ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index):
        %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @softmax_dispatch_0_generic_12x128x128() {
          %c0 = arith.constant 0 : index
          %cst = arith.constant 0.000000e+00 : f32
          %cst_0 = arith.constant -3.40282347E+38 : f32
          %cst_1 = arith.constant 1.000000e+00 : f32
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:12x128x128xf32>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<writeonly:12x128x128xf32>
          %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [12, 128, 128], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:12x128x128xf32> -> tensor<12x128x128xf32>
          %3 = linalg.init_tensor [12, 128, 128] : tensor<12x128x128xf32>
          %4 = linalg.init_tensor [12, 128] : tensor<12x128xf32>
          %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<12x128xf32>) -> tensor<12x128xf32>
          %6 = linalg.fill ins(%cst_0 : f32) outs(%4 : tensor<12x128xf32>) -> tensor<12x128xf32>
          %7 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%2 : tensor<12x128x128xf32>) outs(%6 : tensor<12x128xf32>) {
          ^bb0(%arg0: f32, %arg1: f32):
            %10 = arith.maxf %arg0, %arg1 : f32
            linalg.yield %10 : f32
          } -> tensor<12x128xf32>
          %8:2 = linalg.generic {indexing_maps = [#map0, #map1, #map0, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%2, %7 : tensor<12x128x128xf32>, tensor<12x128xf32>) outs(%3, %5 : tensor<12x128x128xf32>, tensor<12x128xf32>) {
          ^bb0(%arg0: f32, %arg1: f32, %arg2: f32, %arg3: f32):
            %10 = arith.subf %arg0, %arg1 : f32
            %11 = math.exp %10 : f32
            %12 = arith.addf %11, %arg3 : f32
            linalg.yield %11, %12 : f32, f32
          } -> (tensor<12x128x128xf32>, tensor<12x128xf32>)
          %9 = linalg.generic {indexing_maps = [#map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel"]} ins(%8#0, %8#1 : tensor<12x128x128xf32>, tensor<12x128xf32>) outs(%3 : tensor<12x128x128xf32>) {
          ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
            %10 = arith.divf %cst_1, %arg1 : f32
            %11 = arith.mulf %arg0, %10 : f32
            linalg.yield %11 : f32
          } -> tensor<12x128x128xf32>
          flow.dispatch.tensor.store %9, %1, offsets = [0, 0, 0], sizes = [12, 128, 128], strides = [1, 1, 1] : tensor<12x128x128xf32> -> !flow.dispatch.tensor<writeonly:12x128x128xf32>
          return
        }
      }
    }
  }
}


//   CHECK-LABEL:  func.func @conv2d_1x230x230x3_7x7x3x64
//     CHECK-NOT:    vector.transfer_write
//     CHECK-NOT:    vector.transfer_read
//         CHECK:    scf.for
//         CHECK:      scf.for
// CHECK-COUNT-2:        vector.transfer_read
// CHECK-COUNT-4:        vector.contract
//         CHECK:      scf.yield %{{.*}} : vector<4x4xf32>
//         CHECK:    scf.yield %{{.*}} : vector<4x4xf32>
//         CHECK:    vector.transfer_write {{.*}} : vector<4x4xf32>, memref<1x112x112x64xf32>
