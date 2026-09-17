module {
  func.func @op(%input: tensor<1x1x16x4xi8>,
                %w: tensor<1x3x4x1xi8>, %b: tensor<4xi32>,
                %dst: memref<1x1x16x4xi32>) {
    %izp = "tosa.const"() {values = dense<0> : tensor<1xi8>} : () -> tensor<1xi8>
    %wzp = "tosa.const"() {values = dense<0> : tensor<1xi8>} : () -> tensor<1xi8>
    %r = tosa.depthwise_conv2d %input, %w, %b, %izp, %wzp {
      acc_type = i32,
      dilation = array<i64: 1, 1>,
      pad = array<i64: 0, 0, 1, 1>,
      stride = array<i64: 1, 1>
    } : (tensor<1x1x16x4xi8>, tensor<1x3x4x1xi8>, tensor<4xi32>,
         tensor<1xi8>, tensor<1xi8>) -> tensor<1x1x16x4xi32>
    bufferization.materialize_in_destination %r in restrict writable %dst
      : (tensor<1x1x16x4xi32>, memref<1x1x16x4xi32>) -> ()
    return
  }
}