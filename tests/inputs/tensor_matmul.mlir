module {
  func.func @matmul_tensor(%A: tensor<8x8xi16>, %B: tensor<8x8xi16>,
                           %C: memref<8x8xi32>) {
    %init = bufferization.to_tensor %C restrict writable : memref<8x8xi32> to tensor<8x8xi32>
    %0 = linalg.matmul ins(%A, %B: tensor<8x8xi16>, tensor<8x8xi16>)
                       outs(%init: tensor<8x8xi32>) -> tensor<8x8xi32>
    bufferization.materialize_in_destination %0 in restrict writable %C
      : (tensor<8x8xi32>, memref<8x8xi32>) -> ()
    return
  }
}