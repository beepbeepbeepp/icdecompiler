// A is rows 1..3 of a 4x4 buffer: offset 4, contiguous.
func.func @f(%buf: memref<4x4xi16>, %B: memref<4x2xi16>, %C: memref<3x2xi32>) {
  %A = memref.reinterpret_cast %buf to offset: [4], sizes: [3, 4], strides: [4, 1]
      : memref<4x4xi16> to memref<3x4xi16, strided<[4, 1], offset: 4>>
  linalg.matmul ins(%A, %B : memref<3x4xi16, strided<[4, 1], offset: 4>>, memref<4x2xi16>)
                outs(%C : memref<3x2xi32>)
  return
}
