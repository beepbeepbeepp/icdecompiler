// A temporary buffer of booleans: flags[i] = in[i] < 0, then out[i] is in[i]
// or zero. One byte per element in the generated C.
func.func @f(%in: memref<4xi32>, %out: memref<4xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %zero = arith.constant 0 : i32
  %flags = memref.alloc() : memref<4xi1>
  scf.for %i = %c0 to %c4 step %c1 {
    %v = memref.load %in[%i] : memref<4xi32>
    %neg = arith.cmpi slt, %v, %zero : i32
    memref.store %neg, %flags[%i] : memref<4xi1>
  }
  scf.for %i = %c0 to %c4 step %c1 {
    %neg = memref.load %flags[%i] : memref<4xi1>
    %v = memref.load %in[%i] : memref<4xi32>
    %w = arith.select %neg, %zero, %v : i32
    memref.store %w, %out[%i] : memref<4xi32>
  }
  return
}
