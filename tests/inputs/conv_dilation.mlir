// A convolution with a dilated kernel: the taps are two columns apart. The
// im2col rewrite does not handle that, so it stays a convolution on the CPU.
module {
  func.func @conv_dilation(%input: tensor<1x1x8x2xi16>,
                           %w: tensor<4x1x3x2xi16>, %b: tensor<4xi32>,
                           %dst: memref<1x1x8x4xi32>) {
    %izp = "tosa.const"() {values = dense<0> : tensor<1xi16>} : () -> tensor<1xi16>
    %wzp = "tosa.const"() {values = dense<0> : tensor<1xi16>} : () -> tensor<1xi16>

    %acc = tosa.conv2d %input, %w, %b, %izp, %wzp {
      acc_type = i32,
      dilation = array<i64: 1, 2>,
      pad = array<i64: 0, 0, 2, 2>,
      stride = array<i64: 1, 1>
    } : (tensor<1x1x8x2xi16>, tensor<4x1x3x2xi16>, tensor<4xi32>,
         tensor<1xi16>, tensor<1xi16>) -> tensor<1x1x8x4xi32>

    bufferization.materialize_in_destination %acc in restrict writable %dst
      : (tensor<1x1x8x4xi32>, memref<1x1x8x4xi32>) -> ()
    return
  }
}
