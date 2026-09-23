module {
  func.func @matmul_tosa(%A: tensor<1x8x8xf32>, %B: tensor<1x8x8xf32>,
                         %out: memref<1x8x8xf32>) {
    %azp = "tosa.const"() {values = dense<0.0> : tensor<1xf32>} : () -> tensor<1xf32>
    %bzp = "tosa.const"() {values = dense<0.0> : tensor<1xf32>} : () -> tensor<1xf32>
    %0 = tosa.matmul %A, %B, %azp, %bzp
      : (tensor<1x8x8xf32>, tensor<1x8x8xf32>, tensor<1xf32>, tensor<1xf32>)
      -> tensor<1x8x8xf32>
    bufferization.materialize_in_destination %0 in restrict writable %out
      : (tensor<1x8x8xf32>, memref<1x8x8xf32>) -> ()
    return
  }
}