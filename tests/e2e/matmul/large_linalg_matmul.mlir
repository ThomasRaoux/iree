// Test large aligned linalg matmul to make sure we go through the optimized
// path for GPUs.

// Problem size      : 2048x512x1024
// Input type        : F32
// Accumulation type : F32
func.func @matmul_2048x512x1024_f32_f32() {
  %lhs = util.unfoldable_constant dense<1.0> : tensor<2048x1024xf32>
  %rhs = util.unfoldable_constant dense<0.4> : tensor<1024x512xf32>
  %bias = util.unfoldable_constant dense<0.4> : tensor<2048x512xf32>
  %c0 = arith.constant 0.0 : f32
  %init = linalg.init_tensor[2048, 512] : tensor<2048x512xf32>
  %CC = linalg.fill ins(%c0 : f32) outs(%init : tensor<2048x512xf32>) -> tensor<2048x512xf32>
  %D = linalg.matmul ins(%lhs, %rhs: tensor<2048x1024xf32>, tensor<1024x512xf32>)
                    outs(%CC: tensor<2048x512xf32>) -> tensor<2048x512xf32>
  %E = linalg.generic {
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
          iterator_types = ["parallel", "parallel"]}
          ins(%D, %bias : tensor<2048x512xf32>, tensor<2048x512xf32>) outs(%init : tensor<2048x512xf32>) {
        ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):  // no predecessors
          %19 = arith.addf %arg3, %arg4 : f32
          linalg.yield %19 : f32
        } -> (tensor<2048x512xf32>)

  check.expect_almost_eq_const(%E, dense<409.596> : tensor<2048x512xf32>) : tensor<2048x512xf32>
  return
}
