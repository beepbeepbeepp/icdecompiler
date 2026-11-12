// A convolution whose im2col matmul would need K = KH*KW*C = 1*3*43 = 129,
// one more than the accelerator takes. It stays a plain convolution on the CPU.
module {
  func.func @conv_k_limit(%input: tensor<1x1x8x43xi16>,
                          %w: tensor<4x1x3x43xi16>, %b: tensor<4xi32>,
                          %dst: memref<1x1x8x4xi32>) {
    %izp = "tosa.const"() {values = dense<0> : tensor<1xi16>} : () -> tensor<1xi16>
    %wzp = "tosa.const"() {values = dense<0> : tensor<1xi16>} : () -> tensor<1xi16>

    %acc = tosa.conv2d %input, %w, %b, %izp, %wzp {
      acc_type = i32,
      dilation = array<i64: 1, 1>,
      pad = array<i64: 0, 0, 1, 1>,
      stride = array<i64: 1, 1>
    } : (tensor<1x1x8x43xi16>, tensor<4x1x3x43xi16>, tensor<4xi32>,
         tensor<1xi16>, tensor<1xi16>) -> tensor<1x1x8x4xi32>

    bufferization.materialize_in_destination %acc in restrict writable %dst
      : (tensor<1x1x8x4xi32>, memref<1x1x8x4xi32>) -> ()
    return
  }
}
