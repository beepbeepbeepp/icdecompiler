module {
  func.func @two_matmul(%A: memref<8x8xi16>, %B: memref<8x8xi16>,
                        %T: memref<8x8xi32>,
                        %C: memref<8x8xi16>, %D: memref<8x8xi16>,
                        %E: memref<8x8xi32>) {
    // layer 1: T = A * B, akumulator i32
    linalg.matmul ins(%A, %B: memref<8x8xi16>, memref<8x8xi16>)
                  outs(%T: memref<8x8xi32>)

    // layer 2: E = C * D, akumulator i32
    linalg.matmul ins(%C, %D: memref<8x8xi16>, memref<8x8xi16>)
                  outs(%E: memref<8x8xi32>)
    return
  }
}