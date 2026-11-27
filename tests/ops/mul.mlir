module {
  func.func @op(%a: tensor<1x8x4xi16>, %b: tensor<1x8x4xi16>,
                %dst: memref<1x8x4xi32>) {
    %shift = "tosa.const"() {values = dense<0> : tensor<1xi8>} : () -> tensor<1xi8>
    %r = tosa.mul %a, %b, %shift
      : (tensor<1x8x4xi16>, tensor<1x8x4xi16>, tensor<1xi8>) -> tensor<1x8x4xi32>
    bufferization.materialize_in_destination %r in restrict writable %dst
      : (tensor<1x8x4xi32>, memref<1x8x4xi32>) -> ()
    return
  }
}
