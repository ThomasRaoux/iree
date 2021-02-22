// RUN: iree-opt -iree-codegen-convert-to-nvvm %s | IreeFileCheck %s

func @abs_ex_dispatch_0() {
  %0 = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret0, operand_result_index = 2 : i32} : memref<16xf32>
  %1 = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg0, operand_result_index = 0 : i32} : memref<16xf32>
  %2 = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg1, operand_result_index = 1 : i32} : memref<16xf32>
  %3 = "gpu.block_id"() {dimension = "x"} : () -> index
  %4 = "gpu.block_dim"() {dimension = "x"} : () -> index
  %5 = "gpu.thread_id"() {dimension = "x"} : () -> index
  %6 = muli %3, %4 : index
  %7 = addi %6, %5 : index
  %9 = load %1[%7] : memref<16xf32>
  %10 = load %2[%7] : memref<16xf32>
  %11 = addf %9, %10 : f32
  store %11, %0[%7] : memref<16xf32>
  return
}
hal.interface @legacy_io attributes {sym_visibility = "private"} {
  hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
  hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
  hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
}

// CHECK-LABEL: llvm.func @abs_ex_dispatch_0
//  CHECK-SAME: (%{{.*}}: !llvm.ptr<f32>, %{{.*}}: !llvm.ptr<f32>, %{{.*}}: !llvm.ptr<f32>)
//      CHECK:    nvvm.read.ptx.sreg.tid.x
//      CHECK:    llvm.fadd