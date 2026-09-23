#map = affine_map<(d0, d1) -> (d0, d1)>

emitc.include <"systolic_drv.h">

func.func @run_elementwise(%A: memref<8x8xf32>, %B: memref<8x8xf32>, %C: memref<8x8xf32>) {
  linalg.generic {
    indexing_maps = [#map, #map, #map],
    iterator_types = ["parallel", "parallel"]
  } ins(%A, %B : memref<8x8xf32>, memref<8x8xf32>)
    outs(%C : memref<8x8xf32>) {
  ^bb0(%a: f32, %b: f32, %c: f32):
    %0 = arith.mulf %a, %b : f32
    linalg.yield %0 : f32
  }
  return
}