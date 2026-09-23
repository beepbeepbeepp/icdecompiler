module {
  func.func @op(%input: tensor<1x1x16x4xi8>, %dst: memref<1x1x8x4xi8>) {
    %izp = "tosa.const"() {values = dense<0> : tensor<1xi8>} : () -> tensor<1xi8>
    %ozp = "tosa.const"() {values = dense<0> : tensor<1xi8>} : () -> tensor<1xi8>
    %r = tosa.avg_pool2d %input, %izp, %ozp {
      acc_type = i32,
      kernel = array<i64: 1, 2>,
      pad = array<i64: 0, 0, 0, 0>,
      stride = array<i64: 1, 2>
    } : (tensor<1x1x16x4xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x1x8x4xi8>
    bufferization.materialize_in_destination %r in restrict writable %dst
      : (tensor<1x1x8x4xi8>, memref<1x1x8x4xi8>) -> ()
    return
  }
}