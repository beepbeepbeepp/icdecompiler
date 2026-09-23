module {
  func.func @matmul_rect(%A: memref<4x6xi16>, %B: memref<6x8xi16>,
                         %C: memref<4x8xi32>) {
    linalg.matmul ins(%A, %B: memref<4x6xi16>, memref<6x8xi16>)
                  outs(%C: memref<4x8xi32>)
    return
  }
}