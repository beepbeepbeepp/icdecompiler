#map = affine_map<(d0, d1, d2, d3) -> (d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1 floordiv 6 + d2 floordiv 12, d1 mod 6 + (d2 mod 12) floordiv 4, d2 mod 4)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map4 = affine_map<(d0) -> (-d0 + 6, 8)>
#map5 = affine_map<(d0) -> (d0 - 1)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map7 = affine_map<(d0, d1, d2, d3) -> (d2, d3)>
#map8 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
module {
  func.func @conv(%arg0: tensor<1x1x8x4xf32>, %arg1: tensor<8x1x3x4xf32>, %arg2: tensor<8xf32>) -> tensor<1x1x6x8xf32> {
    %0 = tensor.empty() : tensor<1x1x6x8xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2 : tensor<8xf32>) outs(%0 : tensor<1x1x6x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x1x6x8xf32>
    %collapsed = tensor.collapse_shape %arg1 [[0], [1, 2, 3]] : tensor<8x1x3x4xf32> into tensor<8x12xf32>
    %collapsed_0 = tensor.collapse_shape %1 [[0], [1, 2], [3]] : tensor<1x1x6x8xf32> into tensor<1x6x8xf32>
    %2 = tensor.empty() : tensor<1x6x12xf32>
    %3 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x1x8x4xf32>) outs(%2 : tensor<1x6x12xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x6x12xf32>
    %c0 = arith.constant 0 : index
    %c0_1 = arith.constant 0 : index
    %c6 = arith.constant 6 : index
    %c8 = arith.constant 8 : index
    %c8_2 = arith.constant 8 : index
    %c8_3 = arith.constant 8 : index
    %4 = scf.for %arg3 = %c0 to %c6 step %c8_2 iter_args(%arg4 = %collapsed_0) -> (tensor<1x6x8xf32>) {
      %5 = scf.for %arg5 = %c0_1 to %c8 step %c8_3 iter_args(%arg6 = %arg4) -> (tensor<1x6x8xf32>) {
        %c6_4 = arith.constant 6 : index
        %6 = affine.min #map4(%arg3)
        %7 = affine.apply #map5(%6)
        %8 = affine.apply #map5(%6)
        %9 = affine.apply #map5(%6)
        %extracted_slice = tensor.extract_slice %3[0, %arg3, 0] [1, %6, 12] [1, 1, 1] : tensor<1x6x12xf32> to tensor<1x?x12xf32>
        %extracted_slice_5 = tensor.extract_slice %collapsed[%arg5, 0] [8, 12] [1, 1] : tensor<8x12xf32> to tensor<8x12xf32>
        %extracted_slice_6 = tensor.extract_slice %arg6[0, %arg3, %arg5] [1, %6, 8] [1, 1, 1] : tensor<1x6x8xf32> to tensor<1x?x8xf32>
        %10 = linalg.generic {indexing_maps = [#map6, #map7, #map8], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%extracted_slice, %extracted_slice_5 : tensor<1x?x12xf32>, tensor<8x12xf32>) outs(%extracted_slice_6 : tensor<1x?x8xf32>) {
        ^bb0(%in: f32, %in_7: f32, %out: f32):
          %13 = arith.mulf %in, %in_7 : f32
          %14 = arith.addf %13, %out : f32
          linalg.yield %14 : f32
        } -> tensor<1x?x8xf32>
        %11 = affine.apply #map5(%6)
        %12 = affine.apply #map5(%6)
        %inserted_slice = tensor.insert_slice %10 into %arg6[0, %arg3, %arg5] [1, %6, 8] [1, 1, 1] : tensor<1x?x8xf32> into tensor<1x6x8xf32>
        scf.yield %inserted_slice : tensor<1x6x8xf32>
      }
      scf.yield %5 : tensor<1x6x8xf32>
    }
    %expanded = tensor.expand_shape %4 [[0], [1, 2], [3]] output_shape [1, 1, 6, 8] : tensor<1x6x8xf32> into tensor<1x1x6x8xf32>
    return %expanded : tensor<1x1x6x8xf32>
  }
}

