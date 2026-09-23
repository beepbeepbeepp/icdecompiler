module {
  func.func @op(%a: tensor<1x8x4xi8>, %dst: memref<1x4x8xi8>) {
    %r = tosa.transpose %a {perms = array<i32: 0, 2, 1>}
      : (tensor<1x8x4xi8>) -> tensor<1x4x8xi8>
    bufferization.materialize_in_destination %r in restrict writable %dst
      : (tensor<1x4x8xi8>, memref<1x4x8xi8>) -> ()
    return
  }
}