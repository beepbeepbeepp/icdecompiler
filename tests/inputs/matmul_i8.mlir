// An int8 matmul. The accelerator only takes i16 x i16 -> i32, so this stays
// on the CPU and the compiler warns about it.
module {
  func.func @matmul_i8(%A: memref<4x6xi8>, %B: memref<6x8xi8>,
                       %C: memref<4x8xi32>) {
    linalg.matmul ins(%A, %B: memref<4x6xi8>, memref<6x8xi8>)
                  outs(%C: memref<4x8xi32>)
    return
  }
}
