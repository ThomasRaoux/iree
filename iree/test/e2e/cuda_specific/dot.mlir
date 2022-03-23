

func @ksplitmatmul(%arg0 : tensor<2048x65536xf32>, %arg1 : tensor<65536x2048xf32>) -> tensor<2048x2048xf32> {
  %res = "mhlo.dot"(%arg0, %arg1) : (tensor<2048x65536xf32>, tensor<65536x2048xf32>) -> tensor<2048x2048xf32>
  return %res : tensor<2048x2048xf32>
}

//func @ksplitmatmul(%arg0 : tensor<2048x65536xf32>, %arg1 : tensor<65536x2048xf32>) -> tensor<2048x2048xf32> {
//    %cst = arith.constant 0.000000e+00 : f32
//    %7 = tensor.expand_shape %arg1 [[0, 1], [2]] : tensor<65536x2048xf32> into tensor<4x16384x2048xf32>
//    %8 = tensor.expand_shape %arg0 [[0], [1, 2]] : tensor<2048x65536xf32> into tensor<2048x4x16384xf32>
//    %9 = linalg.init_tensor [4, 2048, 2048] : tensor<4x2048x2048xf32>
//    %10 = linalg.fill ins(%cst : f32) outs(%9 : tensor<4x2048x2048xf32>) -> tensor<4x2048x2048xf32>
//    %11 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d1, d0, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%8, %7 : tensor<2048x4x16384xf32>, tensor<4x16384x2048xf32>) outs(%10 : tensor<4x2048x2048xf32>) attrs =  {__internal_linalg_transform__ = "SPLIT"} {
//    ^bb0(%arg5: f32, %arg6: f32, %arg7: f32):
//      %12 = arith.mulf %arg5, %arg6 : f32
//      %13 = arith.addf %arg7, %12 : f32
//      linalg.yield %13 : f32
//    } -> tensor<4x2048x2048xf32>
//    %6 = linalg.init_tensor [2048, 2048] : tensor<2048x2048xf32>
//    %71 = linalg.fill ins(%cst : f32) outs(%6 : tensor<2048x2048xf32>) -> tensor<2048x2048xf32>
//    %81 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%11 : tensor<4x2048x2048xf32>) outs(%71 : tensor<2048x2048xf32>) attrs =  {__internal_linalg_transform__ = "SPLIT"} {
//    ^bb0(%arg4: f32, %arg5: f32):
//      %91 = arith.addf %arg4, %arg5 : f32
//      linalg.yield %91 : f32
//    } -> tensor<2048x2048xf32>
//    return %81 : tensor<2048x2048xf32>
//}

//func @ksplitmatmul(%arg0 : tensor<2048x65536xf32>, %arg1 : tensor<65536x2048xf32>) -> tensor<256x2048x2048xf32>
//{
//  %C0_f = arith.constant 0.0 : f32
//  %A0 = tensor.expand_shape %arg0 [[0], [1, 2]] : tensor<2048x65536xf32> into tensor<2048x256x256xf32>
//  %B0 = tensor.expand_shape %arg1 [[0, 1], [2]] : tensor<65536x2048xf32> into tensor<256x256x2048xf32>
//  %I = linalg.init_tensor [256, 2048, 2048]  : tensor<256x2048x2048xf32>
//  %C0 = linalg.fill ins(%C0_f : f32) outs(%I : tensor<256x2048x2048xf32>) -> tensor<256x2048x2048xf32>
//  %C1 = linalg.generic {
//    indexing_maps = [
//      affine_map<(k0, m, n, k1) -> (m, k0, k1)>,
//      affine_map<(k0, m, n, k1) -> (k0, k1, n)>,
//      affine_map<(k0, m, n, k1) -> (k0, m, n)>
//    ],
//    iterator_types = ["parallel", "parallel", "parallel", "reduction"] }
//      ins(%A0, %B0 : tensor<2048x256x256xf32>, tensor<256x256x2048xf32>)
//     outs(%C0 : tensor<256x2048x2048xf32>) {
//      ^bb(%a: f32, %b: f32, %c: f32) :
//        %d = arith.mulf %a, %b: f32
//        %e = arith.addf %c, %d: f32
//        linalg.yield %e : f32
//  } -> tensor<256x2048x2048xf32>
//  return %C1 : tensor<256x2048x2048xf32>
//}

