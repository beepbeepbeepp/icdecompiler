module {
  func.func @two_calls(%A: memref<8x8xi16>, %B: memref<8x8xi16>,
                       %C: memref<8x8xi32>, %D: memref<8x8xi32>) {
    linalg.matmul ins(%A, %B: memref<8x8xi16>, memref<8x8xi16>)
                  outs(%C: memref<8x8xi32>)
    linalg.matmul ins(%B, %A: memref<8x8xi16>, memref<8x8xi16>)
                  outs(%D: memref<8x8xi32>)
    return
  }
}
