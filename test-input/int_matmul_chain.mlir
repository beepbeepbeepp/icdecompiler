module {
  func.func @matmul_chain(%A: memref<4x4xi16>, %B: memref<4x4xi16>,
                          %C: memref<4x4xi32>,
                          %T: memref<4x4xi16>, %D: memref<4x4xi16>,
                          %E: memref<4x4xi32>) {
    // layer 1: C = A * B
    linalg.matmul ins(%A, %B: memref<4x4xi16>, memref<4x4xi16>)
                  outs(%C: memref<4x4xi32>)

    // layer 2: E = T * D
    linalg.matmul ins(%T, %D: memref<4x4xi16>, memref<4x4xi16>)
                  outs(%E: memref<4x4xi32>)
    return
  }
}