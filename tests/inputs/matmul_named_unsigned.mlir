// linalg.matmul can ask for its inputs to be widened as unsigned. The driver
// multiplies signed values, so this must stay on the CPU.
func.func @f(%A: memref<3x4xi16>, %B: memref<4x2xi16>, %C: memref<3x2xi32>) {
  linalg.matmul {cast = #linalg.type_fn<cast_unsigned>}
    ins(%A, %B : memref<3x4xi16>, memref<4x2xi16>)
    outs(%C : memref<3x2xi32>)
  return
}
