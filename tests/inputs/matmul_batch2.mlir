// A batch matmul with a batch of two. The accelerator takes a single matmul,
// so this stays on the CPU and the compiler warns about it.
module {
  func.func @matmul_batch2(%A: memref<2x4x6xi16>, %B: memref<2x6x8xi16>,
                           %C: memref<2x4x8xi32>) {
    linalg.batch_matmul ins(%A, %B: memref<2x4x6xi16>, memref<2x6x8xi16>)
                        outs(%C: memref<2x4x8xi32>)
    return
  }
}
