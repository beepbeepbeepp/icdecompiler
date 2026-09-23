module {
  func.func @op(%a: tensor<1x8x4xi8>, %dst: memref<1x12x4xi8>) {
    %pad = tosa.const_shape {values = dense<[0, 0, 2, 2, 0, 0]> : tensor<6xindex>}
      : () -> !tosa.shape<6>
    %pv = "tosa.const"() {values = dense<0> : tensor<1xi8>} : () -> tensor<1xi8>
    %r = tosa.pad %a, %pad, %pv
      : (tensor<1x8x4xi8>, !tosa.shape<6>, tensor<1xi8>) -> tensor<1x12x4xi8>
    bufferization.materialize_in_destination %r in restrict writable %dst
      : (tensor<1x12x4xi8>, memref<1x12x4xi8>) -> ()
    return
  }
}