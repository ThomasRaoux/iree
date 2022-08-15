func.func @tensor() {
  %0 = util.unfoldable_constant dense<1.0> : tensor<1307xf32>
  %1 = util.unfoldable_constant dense<2.0> : tensor<1307xf32>
  %result = "mhlo.add"(%0, %1) : (tensor<1307xf32>, tensor<1307xf32>) -> tensor<1307xf32>
  check.expect_almost_eq_const(%result, dense<3.0> : tensor<1307xf32>) : tensor<1307xf32>
  return
}
