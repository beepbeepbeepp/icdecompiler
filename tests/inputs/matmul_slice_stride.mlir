// A is every second row of a 6x4 buffer: not contiguous.
func.func @f(%buf: memref<6x4xi16>, %B: memref<4x2xi16>, %C: memref<3x2xi32>) {
  %A = memref.reinterpret_cast %buf to offset: [0], sizes: [3, 4], strides: [8, 1]
      : memref<6x4xi16> to memref<3x4xi16, strided<[8, 1]>>
  linalg.matmul ins(%A, %B : memref<3x4xi16, strided<[8, 1]>>, memref<4x2xi16>)
                outs(%C : memref<3x2xi32>)
  return
}
