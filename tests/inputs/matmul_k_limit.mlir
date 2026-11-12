// Two matmuls on either side of the accelerator limit: K = 128 is offloaded,
// K = 129 stays on the CPU and the compiler warns about it.
module {
  func.func @matmul_k_limit(%A: memref<2x128xi16>, %B: memref<128x3xi16>,
                            %C: memref<2x3xi32>,
                            %D: memref<2x129xi16>, %E: memref<129x3xi16>,
                            %F: memref<2x3xi32>) {
    linalg.matmul ins(%A, %B: memref<2x128xi16>, memref<128x3xi16>)
                  outs(%C: memref<2x3xi32>)
    linalg.matmul ins(%D, %E: memref<2x129xi16>, memref<129x3xi16>)
                  outs(%F: memref<2x3xi32>)
    return
  }
}
