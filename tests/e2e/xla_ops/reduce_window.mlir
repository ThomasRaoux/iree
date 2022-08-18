

func.func @cumsum_f32() {
  %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = util.unfoldable_constant dense<1.0> : tensor<2x2x2xf32>
  %res = "mhlo.reduce_window"(%1, %0) ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
    %4 = mhlo.add %arg1, %arg2 : tensor<f32>
    "mhlo.return"(%4) : (tensor<f32>) -> ()
  }) {padding = dense<[[1, 0], [0, 0], [0, 0]]> : tensor<3x2xi64>,
      window_dimensions = dense<[2, 1, 1]> : tensor<3xi64>,
      window_strides = dense<1> : tensor<3xi64>
  } : (tensor<2x2x2xf32>, tensor<f32>) -> tensor<2x2x2xf32>
  check.expect_almost_eq_const(%res, dense<[[[1.0, 1.0], [1.0, 1.0]], [[2.0, 2.0], [2.0, 2.0]]]> : tensor<2x2x2xf32>) : tensor<2x2x2xf32>
  return
}
