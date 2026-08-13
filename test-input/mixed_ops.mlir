#map = affine_map<(d0, d1) -> (d0, d1)>

module {
  func.func @mixed(%A: memref<8x8xf32>, %B: memref<8x8xf32>,
                   %C: memref<8x8xf32>, %D: memref<8x8xf32>) {
    // matmul: should go to the systolic array
    linalg.matmul ins(%A, %B: memref<8x8xf32>, memref<8x8xf32>)
                  outs(%C: memref<8x8xf32>)

    // relu-like elementwise: should stay on CPU
    linalg.generic {
      indexing_maps = [#map, #map],
      iterator_types = ["parallel", "parallel"]
    } ins(%C : memref<8x8xf32>) outs(%D : memref<8x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %two = arith.constant 2.0 : f32
      %r = arith.mulf %in, %two : f32
      linalg.yield %r : f32
    }
    return
  }
}