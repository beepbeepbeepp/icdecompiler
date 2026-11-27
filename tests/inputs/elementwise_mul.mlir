// Multiplies two buffers element by element. Nothing is accumulated, so this
// is not multiply-accumulate work: it is not counted in the summary and it
// needs no warning for running on the CPU.

#map = affine_map<(d0, d1) -> (d0, d1)>

func.func @elementwise_mul(%A: memref<8x8xi16>, %B: memref<8x8xi16>,
                           %C: memref<8x8xi16>) {
  linalg.generic {
    indexing_maps = [#map, #map, #map],
    iterator_types = ["parallel", "parallel"]
  } ins(%A, %B : memref<8x8xi16>, memref<8x8xi16>)
    outs(%C : memref<8x8xi16>) {
  ^bb0(%a: i16, %b: i16, %c: i16):
    %0 = arith.muli %a, %b : i16
    linalg.yield %0 : i16
  }
  return
}
