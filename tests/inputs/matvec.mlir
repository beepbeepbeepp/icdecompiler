// A matrix times a vector. It multiplies and accumulates, but the compiler
// only sends a matrix times a matrix to the accelerator.
func.func @f(%A: memref<3x4xi16>, %x: memref<4xi16>, %y: memref<3xi32>) {
  linalg.matvec ins(%A, %x : memref<3x4xi16>, memref<4xi16>)
                outs(%y : memref<3xi32>)
  return
}
