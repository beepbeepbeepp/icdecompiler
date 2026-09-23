module {
  func.func @matmul_requant(%A: memref<8x8xi16>, %B: memref<8x8xi16>,
                            %acc: memref<8x8xi32>, %out: memref<8x8xi16>) {
    // matmul into a 32-bit accumulator
    linalg.matmul ins(%A, %B: memref<8x8xi16>, memref<8x8xi16>)
                  outs(%acc: memref<8x8xi32>)

    // requantize Q16.16 back to Q8.8: shift right by 8, then narrow to i16
    %shift = arith.constant 8 : i32
    linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%acc : memref<8x8xi32>) outs(%out : memref<8x8xi16>) {
    ^bb0(%in: i32, %o: i16):
      %s = arith.shrsi %in, %shift : i32
      %t = arith.trunci %s : i32 to i16
      linalg.yield %t : i16
    }
    return
  }
}