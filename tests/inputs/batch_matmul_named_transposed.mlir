// The same for linalg.batch_matmul: B is stored N x K here.
func.func @f(%A: memref<1x3x4xi16>, %B: memref<1x2x4xi16>, %C: memref<1x3x2xi32>) {
  linalg.batch_matmul indexing_maps = [affine_map<(b, m, n, k) -> (b, m, k)>,
                                       affine_map<(b, m, n, k) -> (b, n, k)>,
                                       affine_map<(b, m, n, k) -> (b, m, n)>]
    ins(%A, %B : memref<1x3x4xi16>, memref<1x2x4xi16>)
    outs(%C : memref<1x3x2xi32>)
  return
}
