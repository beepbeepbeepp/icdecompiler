#map = affine_map<(d0, d1, d2, d3) -> (d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module {
  func.func @conv(%input: memref<1x1x8x4xf32>, %weight: memref<8x1x3x4xf32>,
                  %bias: memref<8xf32>, %output: memref<1x1x6x8xf32>) {
    linalg.generic {indexing_maps = [#map, #map1],
                    iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%bias : memref<8xf32>) outs(%output : memref<1x1x6x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    }
    linalg.conv_2d_nhwc_fhwc {dilations = dense<1> : tensor<2xi64>,
                              strides = dense<1> : tensor<2xi64>}
      ins(%input, %weight : memref<1x1x8x4xf32>, memref<8x1x3x4xf32>)
      outs(%output : memref<1x1x6x8xf32>)
    return
  }
}