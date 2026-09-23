module {
  func.func @op(%a: tensor<1x8x4xf32>, %dst: memref<1x8x4xf32>) {
    %r = tosa.sigmoid %a : (tensor<1x8x4xf32>) -> tensor<1x8x4xf32>
    bufferization.materialize_in_destination %r in restrict writable %dst
      : (tensor<1x8x4xf32>, memref<1x8x4xf32>) -> ()
    return
  }
}