// linalg.matmul can carry its own indexing maps. Here A is stored K x M, so
// this is A^T * B. The driver reads A as M x K and cannot take it as it is.
func.func @f(%A: memref<4x3xi16>, %B: memref<4x2xi16>, %C: memref<3x2xi32>) {
  linalg.matmul indexing_maps = [affine_map<(m, n, k) -> (k, m)>,
                                 affine_map<(m, n, k) -> (k, n)>,
                                 affine_map<(m, n, k) -> (m, n)>]
    ins(%A, %B : memref<4x3xi16>, memref<4x2xi16>)
    outs(%C : memref<3x2xi32>)
  return
}
