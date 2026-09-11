module {
  func.func @conv_q88(%input: tensor<1x1x16x3xi16>,
                      %w: tensor<8x1x5x3xi16>, %b: tensor<8xi32>,
                      %dst: memref<1x1x16x8xi16>) {
    %izp = "tosa.const"() {values = dense<0> : tensor<1xi16>} : () -> tensor<1xi16>
    %wzp = "tosa.const"() {values = dense<0> : tensor<1xi16>} : () -> tensor<1xi16>

    %acc = tosa.conv2d %input, %w, %b, %izp, %wzp {
      acc_type = i32,
      dilation = array<i64: 1, 1>,
      pad = array<i64: 0, 0, 2, 2>,
      stride = array<i64: 1, 1>
    } : (tensor<1x1x16x3xi16>, tensor<8x1x5x3xi16>, tensor<8xi32>,
         tensor<1xi16>, tensor<1xi16>) -> tensor<1x1x16x8xi32>

    // Q16.16 back to Q8.8: multiplier 2^31 with shift 39 is an exact >> 8
    %mul = "tosa.const"() {values = dense<1073741824> : tensor<1xi32>} : () -> tensor<1xi32>
    %shf = "tosa.const"() {values = dense<38> : tensor<1xi8>} : () -> tensor<1xi8>
    %rzp = "tosa.const"() {values = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
    %ozp = "tosa.const"() {values = dense<0> : tensor<1xi16>} : () -> tensor<1xi16>

    %out = tosa.rescale %acc, %mul, %shf, %rzp, %ozp {
      scale32 = true,
      rounding_mode = #tosa.rounding_mode<SINGLE_ROUND>,
      per_channel = false,
      input_unsigned = false, output_unsigned = false
    } : (tensor<1x1x16x8xi32>, tensor<1xi32>, tensor<1xi8>,
         tensor<1xi32>, tensor<1xi16>) -> tensor<1x1x16x8xi16>

    bufferization.materialize_in_destination %out in restrict writable %dst
      : (tensor<1x1x16x8xi16>, memref<1x1x16x8xi16>) -> ()
    return
  }
}