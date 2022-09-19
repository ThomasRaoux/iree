
func.func @elementwise(
  %A : tensor<16xf32>, %B : tensor<16xf32>, %C: tensor<16xf32>) -> tensor<16xf32> {

  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], 
    iterator_types = ["parallel"]}
    ins(%A, %B : tensor<16xf32>, tensor<16xf32>) outs(%C : tensor<16xf32>) {
  ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):  // no predecessors
    %1 = arith.addf %arg0, %arg1 : f32
    %2 = arith.addf %1, %arg2 : f32
    linalg.yield %2 : f32
   } -> tensor<16xf32>

  return %0 : tensor<16xf32>
}
