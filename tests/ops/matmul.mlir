module {
  func.func @op(%a: tensor<1x8x16xi16>, %b: tensor<1x16x4xi16>,
                %dst: memref<1x8x4xi32>) {
    %azp = "tosa.const"() {values = dense<0> : tensor<1xi16>} : () -> tensor<1xi16>
    %bzp = "tosa.const"() {values = dense<0> : tensor<1xi16>} : () -> tensor<1xi16>
    %r = tosa.matmul %a, %b, %azp, %bzp
      : (tensor<1x8x16xi16>, tensor<1x16x4xi16>, tensor<1xi16>, tensor<1xi16>) -> tensor<1x8x4xi32>
    bufferization.materialize_in_destination %r in restrict writable %dst
      : (tensor<1x8x4xi32>, memref<1x8x4xi32>) -> ()
    return
  }
}
