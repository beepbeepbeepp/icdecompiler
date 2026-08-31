#map = affine_map<(d0, d1, d2, d3) -> (d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1 floordiv 6 + d2 floordiv 12, d1 mod 6 + (d2 mod 12) floordiv 4, d2 mod 4)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d2, d3)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
module attributes {transform.with_named_sequence} {
  func.func @conv(%arg0: tensor<1x1x8x4xf32>, %arg1: tensor<8x1x3x4xf32>, %arg2: tensor<8xf32>, %dst: memref<1x1x6x8xf32>) {
    %0 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2 = tensor.empty() : tensor<1x1x6x8xf32>
    %3 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2 : tensor<8xf32>) outs(%2 : tensor<1x1x6x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x1x6x8xf32>
    %collapsed = tensor.collapse_shape %arg1 [[0], [1, 2, 3]] : tensor<8x1x3x4xf32> into tensor<8x12xf32>
    %collapsed_0 = tensor.collapse_shape %3 [[0], [1, 2], [3]] : tensor<1x1x6x8xf32> into tensor<1x6x8xf32>
    %4 = tensor.empty() : tensor<1x6x12xf32>
    %5 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x1x8x4xf32>) outs(%4 : tensor<1x6x12xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x6x12xf32>
    %6 = linalg.generic {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%5, %collapsed : tensor<1x6x12xf32>, tensor<8x12xf32>) outs(%collapsed_0 : tensor<1x6x8xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %7 = arith.mulf %in, %in_1 : f32
      %8 = arith.addf %7, %out : f32
      linalg.yield %8 : f32
    } -> tensor<1x6x8xf32>
    %expanded = tensor.expand_shape %6 [[0], [1, 2], [3]] output_shape [1, 1, 6, 8] : tensor<1x6x8xf32> into tensor<1x1x6x8xf32>
    bufferization.materialize_in_destination %expanded in restrict writable %dst : (tensor<1x1x6x8xf32>, memref<1x1x6x8xf32>) -> ()
    return
  }
}

