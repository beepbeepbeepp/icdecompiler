// A is fine. B is a static view of a buffer whose own shape is dynamic, so
// the rewrite cannot turn B into a C pointer, and it only finds that out
// after it has already built the constants and the pointer for A.
func.func @f(%A: memref<3x4xi16>, %Bdyn: memref<?x?xi16>, %C: memref<3x2xi32>) {
  %B = memref.reinterpret_cast %Bdyn to offset: [0], sizes: [4, 2], strides: [2, 1]
      : memref<?x?xi16> to memref<4x2xi16>
  linalg.matmul ins(%A, %B : memref<3x4xi16>, memref<4x2xi16>)
                outs(%C : memref<3x2xi32>)
  return
}
