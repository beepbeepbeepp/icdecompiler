func.func @f(%A: memref<3x4xi16>, %B: memref<4x2xi16>, %C: memref<3x2xi32>) {
  linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                     affine_map<(d0, d1, d2) -> (d2, d1)>,
                     affine_map<(d0, d1, d2) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%A, %B : memref<3x4xi16>, memref<4x2xi16>)
    outs(%C : memref<3x2xi32>) {
  ^bb0(%a: i16, %b: i16, %c: i32):
    %x = arith.extsi %a : i16 to i32
    %0 = arith.muli %x, %x : i32
    %1 = arith.addi %0, %c : i32
    linalg.yield %1 : i32
  }
  return
}
