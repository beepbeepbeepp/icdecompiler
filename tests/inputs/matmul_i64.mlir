// A matmul that accumulates in 64 bits: 3 x 4 x 2 = 24 multiply-accumulates.
// The summary used to tell a MAC from a requantisation multiply by its width
// and reported none here.
func.func @f(%A: memref<3x4xi32>, %B: memref<4x2xi32>, %C: memref<3x2xi64>) {
  linalg.matmul ins(%A, %B : memref<3x4xi32>, memref<4x2xi32>)
                outs(%C : memref<3x2xi64>)
  return
}
