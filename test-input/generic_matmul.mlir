#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>

emitc.include <"systolic_drv.h">

func.func @run_generic(%A: memref<8x8xf32>, %B: memref<8x8xf32>, %C: memref<8x8xf32>) {
  linalg.generic {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%A, %B : memref<8x8xf32>, memref<8x8xf32>)
    outs(%C : memref<8x8xf32>) {
  ^bb0(%a: f32, %b: f32, %c: f32):
    %0 = arith.mulf %a, %b : f32
    %1 = arith.addf %0, %c : f32
    linalg.yield %1 : f32
  }
  return
}