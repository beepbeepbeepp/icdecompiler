module {
  func.func @conv(%input: tensor<1x1x8x4xf32>, %weight: tensor<8x1x3x4xf32>,
                  %bias: tensor<8xf32>, %out: memref<1x1x6x8xf32>) {
    %izp = "tosa.const"() {values = dense<0.0> : tensor<1xf32>} : () -> tensor<1xf32>
    %wzp = "tosa.const"() {values = dense<0.0> : tensor<1xf32>} : () -> tensor<1xf32>
    %0 = tosa.conv2d %input, %weight, %bias, %izp, %wzp {
      acc_type = f32,
      dilation = array<i64: 1, 1>,
      pad = array<i64: 0, 0, 0, 0>,
      stride = array<i64: 1, 1>
    } : (tensor<1x1x8x4xf32>, tensor<8x1x3x4xf32>, tensor<8xf32>,
         tensor<1xf32>, tensor<1xf32>) -> tensor<1x1x6x8xf32>
    bufferization.materialize_in_destination %0 in restrict writable %out
      : (tensor<1x1x6x8xf32>, memref<1x1x6x8xf32>) -> ()
    return
  }
}