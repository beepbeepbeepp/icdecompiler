module {
  func.func @har_model(
      %input: tensor<1x1x512x3xf32>,
      %w1: tensor<16x1x15x3xf32>, %b1: tensor<16xf32>,
      %w2: tensor<32x1x7x16xf32>, %b2: tensor<32xf32>,
      %w3: tensor<64x1x3x32xf32>, %b3: tensor<64xf32>,
      %wf1: tensor<1x64x32xf32>,  %bf1: tensor<32xf32>,
      %wf2: tensor<1x32x4xf32>,   %bf2: tensor<4xf32>,
      %dst: memref<1x1x4xf32>) {
    %zp = "tosa.const"() {values = dense<0.0> : tensor<1xf32>} : () -> tensor<1xf32>

    // block 1: conv1d k=15 s=2 pad same, relu, maxpool 2
    %c1 = tosa.conv2d %input, %w1, %b1, %zp, %zp {
      acc_type = f32, dilation = array<i64: 1, 1>,
      pad = array<i64: 0, 0, 6, 7>, stride = array<i64: 1, 2>
    } : (tensor<1x1x512x3xf32>, tensor<16x1x15x3xf32>, tensor<16xf32>,
         tensor<1xf32>, tensor<1xf32>) -> tensor<1x1x256x16xf32>
    %r1 = tosa.clamp %c1 {min_val = 0.0 : f32, max_val = 3.4028235e+38 : f32}
      : (tensor<1x1x256x16xf32>) -> tensor<1x1x256x16xf32>
    %p1 = tosa.max_pool2d %r1 {
      kernel = array<i64: 1, 2>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 2>
    } : (tensor<1x1x256x16xf32>) -> tensor<1x1x128x16xf32>

    // block 2: conv1d k=7 s=2 pad same, relu, maxpool 2
    %c2 = tosa.conv2d %p1, %w2, %b2, %zp, %zp {
      acc_type = f32, dilation = array<i64: 1, 1>,
      pad = array<i64: 0, 0, 2, 3>, stride = array<i64: 1, 2>
    } : (tensor<1x1x128x16xf32>, tensor<32x1x7x16xf32>, tensor<32xf32>,
         tensor<1xf32>, tensor<1xf32>) -> tensor<1x1x64x32xf32>
    %r2 = tosa.clamp %c2 {min_val = 0.0 : f32, max_val = 3.4028235e+38 : f32}
      : (tensor<1x1x64x32xf32>) -> tensor<1x1x64x32xf32>
    %p2 = tosa.max_pool2d %r2 {
      kernel = array<i64: 1, 2>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 2>
    } : (tensor<1x1x64x32xf32>) -> tensor<1x1x32x32xf32>

    // block 3: conv1d k=3 s=1 pad same, relu
    %c3 = tosa.conv2d %p2, %w3, %b3, %zp, %zp {
      acc_type = f32, dilation = array<i64: 1, 1>,
      pad = array<i64: 0, 0, 1, 1>, stride = array<i64: 1, 1>
    } : (tensor<1x1x32x32xf32>, tensor<64x1x3x32xf32>, tensor<64xf32>,
         tensor<1xf32>, tensor<1xf32>) -> tensor<1x1x32x64xf32>
    %r3 = tosa.clamp %c3 {min_val = 0.0 : f32, max_val = 3.4028235e+38 : f32}
      : (tensor<1x1x32x64xf32>) -> tensor<1x1x32x64xf32>

    // global sum over the length axis; the 1/32 factor is folded into wf1
    %s3 = tosa.reduce_sum %r3 {axis = 2 : i32}
      : (tensor<1x1x32x64xf32>) -> tensor<1x1x1x64xf32>
    %gs = tosa.const_shape {values = dense<[1, 1, 64]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %flat = tosa.reshape %s3, %gs
      : (tensor<1x1x1x64xf32>, !tosa.shape<3>) -> tensor<1x1x64xf32>

    // dense 64 -> 32, relu
    %m1 = tosa.matmul %flat, %wf1, %zp, %zp
      : (tensor<1x1x64xf32>, tensor<1x64x32xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x1x32xf32>
    %bs1 = tosa.const_shape {values = dense<[1, 1, 32]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %bb1 = tosa.reshape %bf1, %bs1 : (tensor<32xf32>, !tosa.shape<3>) -> tensor<1x1x32xf32>
    %a1 = tosa.add %m1, %bb1
      : (tensor<1x1x32xf32>, tensor<1x1x32xf32>) -> tensor<1x1x32xf32>
    %rf1 = tosa.clamp %a1 {min_val = 0.0 : f32, max_val = 3.4028235e+38 : f32}
      : (tensor<1x1x32xf32>) -> tensor<1x1x32xf32>

    // dense 32 -> 4
    %m2 = tosa.matmul %rf1, %wf2, %zp, %zp
      : (tensor<1x1x32xf32>, tensor<1x32x4xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x1x4xf32>
    %bs2 = tosa.const_shape {values = dense<[1, 1, 4]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %bb2 = tosa.reshape %bf2, %bs2 : (tensor<4xf32>, !tosa.shape<3>) -> tensor<1x1x4xf32>
    %logits = tosa.add %m2, %bb2
      : (tensor<1x1x4xf32>, tensor<1x1x4xf32>) -> tensor<1x1x4xf32>

    bufferization.materialize_in_destination %logits in restrict writable %dst
      : (tensor<1x1x4xf32>, memref<1x1x4xf32>) -> ()
    return
  }
}