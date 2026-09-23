module {
  func.func @rescale_test(%acc: tensor<1x8x4xi32>, %dst: memref<1x8x4xi8>) {
    %izp = "tosa.const"() {values = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
    %ozp = "tosa.const"() {values = dense<-128> : tensor<1xi8>} : () -> tensor<1xi8>
    %mul = "tosa.const"() {values = dense<1112281741> : tensor<4xi32>} : () -> tensor<4xi32>
    %shf = "tosa.const"() {values = dense<38> : tensor<4xi8>} : () -> tensor<4xi8>

    %out = tosa.rescale %acc, %mul, %shf, %izp, %ozp {
      scale32 = true,
      rounding_mode = #tosa.rounding_mode<DOUBLE_ROUND>,
      per_channel = true,
      input_unsigned = false,
      output_unsigned = false
    } : (tensor<1x8x4xi32>, tensor<4xi32>, tensor<4xi8>, tensor<1xi32>, tensor<1xi8>)
      -> tensor<1x8x4xi8>

    bufferization.materialize_in_destination %out in restrict writable %dst
      : (tensor<1x8x4xi8>, memref<1x8x4xi8>) -> ()
    return
  }
}