// A matmul whose first operand has a zero point: (a - 3) * b. The accelerator
// multiplies the values as they are.
func.func @f(%A: memref<3x4xi16>, %B: memref<4x2xi16>, %C: memref<3x2xi32>) {
  %azp = arith.constant 3 : i32
  %bzp = arith.constant 0 : i32
  linalg.quantized_matmul
    ins(%A, %B, %azp, %bzp : memref<3x4xi16>, memref<4x2xi16>, i32, i32)
    outs(%C : memref<3x2xi32>)
  return
}
