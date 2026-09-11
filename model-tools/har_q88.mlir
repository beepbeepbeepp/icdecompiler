module {
  func.func @har_model(%input1: tensor<1x1x512x3xi16>, %w2: tensor<16x1x15x3xi16>, %b3: tensor<16xi32>, %w14: tensor<32x1x7x16xi16>, %b15: tensor<32xi32>, %w26: tensor<64x1x3x32xi16>, %b27: tensor<64xi32>, %w46: tensor<1x64x32xi16>, %b47: tensor<32xi32>, %w60: tensor<1x32x4xi16>, %b61: tensor<4xi32>, %dst73: memref<1x1x4xi16>) {
    %c4 = "tosa.const"() {values = dense<0> : tensor<1xi16>} : () -> tensor<1xi16>
    %c5 = "tosa.const"() {values = dense<0> : tensor<1xi16>} : () -> tensor<1xi16>
    %a6 = tosa.conv2d %input1, %w2, %b3, %c4, %c5 { acc_type = i32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 6, 7>, stride = array<i64: 1, 2> } : (tensor<1x1x512x3xi16>, tensor<16x1x15x3xi16>, tensor<16xi32>, tensor<1xi16>, tensor<1xi16>) -> tensor<1x1x256x16xi32>
    %c7 = "tosa.const"() {values = dense<1073741824> : tensor<1xi32>} : () -> tensor<1xi32>
    %c8 = "tosa.const"() {values = dense<38> : tensor<1xi8>} : () -> tensor<1xi8>
    %c9 = "tosa.const"() {values = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
    %c10 = "tosa.const"() {values = dense<0> : tensor<1xi16>} : () -> tensor<1xi16>
    %r11 = tosa.rescale %a6, %c7, %c8, %c9, %c10 { scale32 = true, rounding_mode = #tosa.rounding_mode<SINGLE_ROUND>, per_channel = false, input_unsigned = false, output_unsigned = false } : (tensor<1x1x256x16xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi16>) -> tensor<1x1x256x16xi16>
    %v12 = tosa.clamp %r11 { min_val = 0 : i16, max_val = 32767 : i16 } : (tensor<1x1x256x16xi16>) -> tensor<1x1x256x16xi16>
    %v13 = tosa.max_pool2d %v12 { kernel = array<i64: 1, 2>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 2> } : (tensor<1x1x256x16xi16>) -> tensor<1x1x128x16xi16>
    %c16 = "tosa.const"() {values = dense<0> : tensor<1xi16>} : () -> tensor<1xi16>
    %c17 = "tosa.const"() {values = dense<0> : tensor<1xi16>} : () -> tensor<1xi16>
    %a18 = tosa.conv2d %v13, %w14, %b15, %c16, %c17 { acc_type = i32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 2, 3>, stride = array<i64: 1, 2> } : (tensor<1x1x128x16xi16>, tensor<32x1x7x16xi16>, tensor<32xi32>, tensor<1xi16>, tensor<1xi16>) -> tensor<1x1x64x32xi32>
    %c19 = "tosa.const"() {values = dense<1073741824> : tensor<1xi32>} : () -> tensor<1xi32>
    %c20 = "tosa.const"() {values = dense<38> : tensor<1xi8>} : () -> tensor<1xi8>
    %c21 = "tosa.const"() {values = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
    %c22 = "tosa.const"() {values = dense<0> : tensor<1xi16>} : () -> tensor<1xi16>
    %r23 = tosa.rescale %a18, %c19, %c20, %c21, %c22 { scale32 = true, rounding_mode = #tosa.rounding_mode<SINGLE_ROUND>, per_channel = false, input_unsigned = false, output_unsigned = false } : (tensor<1x1x64x32xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi16>) -> tensor<1x1x64x32xi16>
    %v24 = tosa.clamp %r23 { min_val = 0 : i16, max_val = 32767 : i16 } : (tensor<1x1x64x32xi16>) -> tensor<1x1x64x32xi16>
    %v25 = tosa.max_pool2d %v24 { kernel = array<i64: 1, 2>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 2> } : (tensor<1x1x64x32xi16>) -> tensor<1x1x32x32xi16>
    %c28 = "tosa.const"() {values = dense<0> : tensor<1xi16>} : () -> tensor<1xi16>
    %c29 = "tosa.const"() {values = dense<0> : tensor<1xi16>} : () -> tensor<1xi16>
    %a30 = tosa.conv2d %v25, %w26, %b27, %c28, %c29 { acc_type = i32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 1, 1>, stride = array<i64: 1, 1> } : (tensor<1x1x32x32xi16>, tensor<64x1x3x32xi16>, tensor<64xi32>, tensor<1xi16>, tensor<1xi16>) -> tensor<1x1x32x64xi32>
    %c31 = "tosa.const"() {values = dense<1073741824> : tensor<1xi32>} : () -> tensor<1xi32>
    %c32 = "tosa.const"() {values = dense<38> : tensor<1xi8>} : () -> tensor<1xi8>
    %c33 = "tosa.const"() {values = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
    %c34 = "tosa.const"() {values = dense<0> : tensor<1xi16>} : () -> tensor<1xi16>
    %r35 = tosa.rescale %a30, %c31, %c32, %c33, %c34 { scale32 = true, rounding_mode = #tosa.rounding_mode<SINGLE_ROUND>, per_channel = false, input_unsigned = false, output_unsigned = false } : (tensor<1x1x32x64xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi16>) -> tensor<1x1x32x64xi16>
    %v36 = tosa.clamp %r35 { min_val = 0 : i16, max_val = 32767 : i16 } : (tensor<1x1x32x64xi16>) -> tensor<1x1x32x64xi16>
    %v37 = tosa.cast %v36 : (tensor<1x1x32x64xi16>) -> tensor<1x1x32x64xi32>
    %a38 = tosa.reduce_sum %v37 {axis = 2 : i32} : (tensor<1x1x32x64xi32>) -> tensor<1x1x1x64xi32>
    %c39 = "tosa.const"() {values = dense<67108864> : tensor<1xi32>} : () -> tensor<1xi32>
    %c40 = "tosa.const"() {values = dense<31> : tensor<1xi8>} : () -> tensor<1xi8>
    %c41 = "tosa.const"() {values = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
    %c42 = "tosa.const"() {values = dense<0> : tensor<1xi16>} : () -> tensor<1xi16>
    %r43 = tosa.rescale %a38, %c39, %c40, %c41, %c42 { scale32 = true, rounding_mode = #tosa.rounding_mode<SINGLE_ROUND>, per_channel = false, input_unsigned = false, output_unsigned = false } : (tensor<1x1x1x64xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi16>) -> tensor<1x1x1x64xi16>
    %s44 = tosa.const_shape {values = dense<[1, 1, 64]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %v45 = tosa.reshape %r43, %s44 : (tensor<1x1x1x64xi16>, !tosa.shape<3>) -> tensor<1x1x64xi16>
    %c48 = "tosa.const"() {values = dense<0> : tensor<1xi16>} : () -> tensor<1xi16>
    %c49 = "tosa.const"() {values = dense<0> : tensor<1xi16>} : () -> tensor<1xi16>
    %a50 = tosa.matmul %v45, %w46, %c48, %c49 : (tensor<1x1x64xi16>, tensor<1x64x32xi16>, tensor<1xi16>, tensor<1xi16>) -> tensor<1x1x32xi32>
    %s51 = tosa.const_shape {values = dense<[1, 1, 32]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %v52 = tosa.reshape %b47, %s51 : (tensor<32xi32>, !tosa.shape<3>) -> tensor<1x1x32xi32>
    %a53 = tosa.add %a50, %v52 : (tensor<1x1x32xi32>, tensor<1x1x32xi32>) -> tensor<1x1x32xi32>
    %c54 = "tosa.const"() {values = dense<1073741824> : tensor<1xi32>} : () -> tensor<1xi32>
    %c55 = "tosa.const"() {values = dense<38> : tensor<1xi8>} : () -> tensor<1xi8>
    %c56 = "tosa.const"() {values = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
    %c57 = "tosa.const"() {values = dense<0> : tensor<1xi16>} : () -> tensor<1xi16>
    %r58 = tosa.rescale %a53, %c54, %c55, %c56, %c57 { scale32 = true, rounding_mode = #tosa.rounding_mode<SINGLE_ROUND>, per_channel = false, input_unsigned = false, output_unsigned = false } : (tensor<1x1x32xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi16>) -> tensor<1x1x32xi16>
    %v59 = tosa.clamp %r58 { min_val = 0 : i16, max_val = 32767 : i16 } : (tensor<1x1x32xi16>) -> tensor<1x1x32xi16>
    %c62 = "tosa.const"() {values = dense<0> : tensor<1xi16>} : () -> tensor<1xi16>
    %c63 = "tosa.const"() {values = dense<0> : tensor<1xi16>} : () -> tensor<1xi16>
    %a64 = tosa.matmul %v59, %w60, %c62, %c63 : (tensor<1x1x32xi16>, tensor<1x32x4xi16>, tensor<1xi16>, tensor<1xi16>) -> tensor<1x1x4xi32>
    %s65 = tosa.const_shape {values = dense<[1, 1, 4]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %v66 = tosa.reshape %b61, %s65 : (tensor<4xi32>, !tosa.shape<3>) -> tensor<1x1x4xi32>
    %a67 = tosa.add %a64, %v66 : (tensor<1x1x4xi32>, tensor<1x1x4xi32>) -> tensor<1x1x4xi32>
    %c68 = "tosa.const"() {values = dense<1073741824> : tensor<1xi32>} : () -> tensor<1xi32>
    %c69 = "tosa.const"() {values = dense<38> : tensor<1xi8>} : () -> tensor<1xi8>
    %c70 = "tosa.const"() {values = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
    %c71 = "tosa.const"() {values = dense<0> : tensor<1xi16>} : () -> tensor<1xi16>
    %r72 = tosa.rescale %a67, %c68, %c69, %c70, %c71 { scale32 = true, rounding_mode = #tosa.rounding_mode<SINGLE_ROUND>, per_channel = false, input_unsigned = false, output_unsigned = false } : (tensor<1x1x4xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi16>) -> tensor<1x1x4xi16>
    bufferization.materialize_in_destination %r72 in restrict writable %dst73 : (tensor<1x1x4xi16>, memref<1x1x4xi16>) -> ()
    return
  }
}
