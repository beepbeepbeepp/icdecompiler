// A depthwise convolution with the element types the accelerator takes. Each
// channel has its own kernel, so it is not one matmul and stays on the CPU.
module {
  func.func @conv_depthwise(%input: tensor<1x1x16x4xi16>,
                            %w: tensor<1x3x4x1xi16>, %b: tensor<4xi32>,
                            %dst: memref<1x1x16x4xi32>) {
    %izp = "tosa.const"() {values = dense<0> : tensor<1xi16>} : () -> tensor<1xi16>
    %wzp = "tosa.const"() {values = dense<0> : tensor<1xi16>} : () -> tensor<1xi16>
    %r = tosa.depthwise_conv2d %input, %w, %b, %izp, %wzp {
      acc_type = i32,
      dilation = array<i64: 1, 1>,
      pad = array<i64: 0, 0, 1, 1>,
      stride = array<i64: 1, 1>
    } : (tensor<1x1x16x4xi16>, tensor<1x3x4x1xi16>, tensor<4xi32>,
         tensor<1xi16>, tensor<1xi16>) -> tensor<1x1x16x4xi32>
    bufferization.materialize_in_destination %r in restrict writable %dst
      : (tensor<1x1x16x4xi32>, memref<1x1x16x4xi32>) -> ()
    return
  }
}
