module {
  func.func @two_layer(%input: tensor<1x1x8x4xf32>,
                       %w1: tensor<8x1x3x4xf32>, %b1: tensor<8xf32>,
                       %w2: tensor<4x1x3x8xf32>, %b2: tensor<4xf32>,
                       %dst: memref<1x1x4x4xf32>) {
    %izp = "tosa.const"() {values = dense<0.0> : tensor<1xf32>} : () -> tensor<1xf32>
    %wzp = "tosa.const"() {values = dense<0.0> : tensor<1xf32>} : () -> tensor<1xf32>

    %0 = tosa.conv2d %input, %w1, %b1, %izp, %wzp {
      acc_type = f32, dilation = array<i64: 1, 1>,
      pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>
    } : (tensor<1x1x8x4xf32>, tensor<8x1x3x4xf32>, tensor<8xf32>,
         tensor<1xf32>, tensor<1xf32>) -> tensor<1x1x6x8xf32>

    %1 = tosa.conv2d %0, %w2, %b2, %izp, %wzp {
      acc_type = f32, dilation = array<i64: 1, 1>,
      pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>
    } : (tensor<1x1x6x8xf32>, tensor<4x1x3x8xf32>, tensor<4xf32>,
         tensor<1xf32>, tensor<1xf32>) -> tensor<1x1x4x4xf32>

    bufferization.materialize_in_destination %1 in restrict writable %dst
      : (tensor<1x1x4x4xf32>, memref<1x1x4x4xf32>) -> ()
    return
  }
}