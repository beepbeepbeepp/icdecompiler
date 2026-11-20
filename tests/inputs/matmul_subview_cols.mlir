// A is the first 4 columns of a 3x8 buffer: rows are 8 apart, not 4.
func.func @f(%buf: memref<3x8xi16>, %B: memref<4x2xi16>, %C: memref<3x2xi32>) {
  %A = memref.subview %buf[0, 0] [3, 4] [1, 1]
      : memref<3x8xi16> to memref<3x4xi16, strided<[8, 1]>>
  linalg.matmul ins(%A, %B : memref<3x4xi16, strided<[8, 1]>>, memref<4x2xi16>)
                outs(%C : memref<3x2xi32>)
  return
}
