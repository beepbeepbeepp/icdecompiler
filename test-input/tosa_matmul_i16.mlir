module {
  func.func @batch_matmul_i16(%A: memref<1x8x8xi16>, %B: memref<1x8x8xi16>,
                              %C: memref<1x8x8xi32>) {
    linalg.batch_matmul ins(%A, %B: memref<1x8x8xi16>, memref<1x8x8xi16>)
                        outs(%C: memref<1x8x8xi32>)
    return
  }
}