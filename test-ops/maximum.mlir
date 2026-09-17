module {
  func.func @op(%a: tensor<1x8x4xi8>, %b: tensor<1x8x4xi8>,
                %dst: memref<1x8x4xi8>) {
    %r = tosa.maximum %a, %b : (tensor<1x8x4xi8>, tensor<1x8x4xi8>) -> tensor<1x8x4xi8>
    bufferization.materialize_in_destination %r in restrict writable %dst
      : (tensor<1x8x4xi8>, memref<1x8x4xi8>) -> ()
    return
  }
}