// RUN: iree-opt %s -allow-unregistered-dialect -iree-llvmgpu-vector-to-gpu -canonicalize -split-input-file | FileCheck %s

// CHECK-LABEL: func @copies_to_asyncs
func @copies_to_asyncs(%a: memref<1024x1024xf32>) {
  %0 = memref.alloc() : memref<4x32x16xf32, 3>
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %cst_0 = arith.constant 0.000000e+00 : f32
  // CHECK: %[[CP0:.*]] = gpu.device_async_copy {{.*}}, {{.*}}, 4
  %1 = vector.transfer_read %a[%c0, %c0], %cst_0 {in_bounds = [true]} : memref<1024x1024xf32>, vector<4xf32>
  vector.transfer_write %1, %0[%c0, %c0, %c0] {in_bounds = [true]} : vector<4xf32>, memref<4x32x16xf32, 3>
  // CHECK-NOT: gpu.device_async_create_group

  // CHECK: %[[CP1:.*]] = gpu.device_async_copy {{.*}}, {{.*}}, 1
  %2 = vector.transfer_read %a[%c0, %c4], %cst_0 {in_bounds = [true]} : memref<1024x1024xf32>, vector<1xf32>
  vector.transfer_write %2, %0[%c0, %c4, %c0] {in_bounds = [true]} : vector<1xf32>, memref<4x32x16xf32, 3>
  // CHECK: %[[G:.*]] = gpu.device_async_create_group %[[CP0]], %[[CP1]]
  // CHECK: gpu.device_async_wait %[[G]]
  return
}

// -----

func @ksplitmatmul_basic(%a: memref<128x16x256xf32>) -> vector<16x1x8xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %0 = vector.transfer_read %a[%c2, %c3, %c4], %cst {in_bounds = [true, true, true]} : memref<128x16x256xf32>, vector<16x1x8xf32>
  return %0 : vector<16x1x8xf32>
}
// CHECK-LABEL: func @ksplitmatmul_basic
//   CHECK-DAG: %[[ID:.*]] = arith.constant 35 : index
//   CHECK-DAG: %[[ID2:.*]] = arith.constant 4 : index  
//   CHECK-DAG: %[[CST:.*]] = arith.constant 0.000000e+00 : f32
//       CHECK: %[[M:.*]] = memref.collapse_shape %{{.*}}[0, 1], [2]] : memref<128x16x256xf32> into memref<2048x256xf32>
//       CHECK: vector.transfer_read %[[M]][%[[ID]], %[[ID2]]]
//  CHECK-SAME: {in_bounds = [true, true]} : memref<2048x256xf32>, vector<16x8xf32>
//       CHECK: vector.broadcast %{{.*}} : vector<16x8xf32> to vector<1x16x8xf32>
//       CHECK: vector.transpose %{{.*}} [1, 0, 2] : vector<1x16x8xf32> to vector<16x1x8xf32>
//       CHECK: return %{{.*}} : vector<16x1x8xf32>

// -----

func @ksplitmatmul_nounitdim(%a: memref<128x16x256xf32>) -> vector<16x2x8xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %0 = vector.transfer_read %a[%c2, %c3, %c4], %cst {in_bounds = [true, true, true]} : memref<128x16x256xf32>, vector<16x2x8xf32>
  return %0 : vector<16x2x8xf32>
}
// CHECK-LABEL: func @ksplitmatmul_nounitdim
//   CHECK-DAG: %[[ID:.*]] = arith.constant 2 : index
//   CHECK-DAG: %[[ID2:.*]] = arith.constant 3 : index
//   CHECK-DAG: %[[ID3:.*]] = arith.constant 4 : index    
//   CHECK-DAG: %[[CST:.*]] = arith.constant 0.000000e+00 : f32
//       CHECK: vector.transfer_read %{{.*}}[%[[ID]], %[[ID2]], %[[ID3]]]
//  CHECK-SAME: {in_bounds = [true, true, true]} : memref<128x16x256xf32>, vector<16x2x8xf32>
//       CHECK: return %{{.*}} : vector<16x2x8xf32>

