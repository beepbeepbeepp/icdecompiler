// A declaration without a body next to the function to compile. There is
// nothing to generate for it, and it must not stop the compiler.
func.func private @unused(memref<4xi32>)
func.func @f(%in: memref<4xi32>, %out: memref<4xi32>) {
  memref.copy %in, %out : memref<4xi32> to memref<4xi32>
  return
}
