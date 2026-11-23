// A function with a branch between blocks. %t1 is filled in the first block
// and read in a later one, after %t2 has been filled there. The arena pass
// reads lifetimes off one block only and would give both the same memory, so
// the compiler refuses the function instead.
func.func @two_blocks(%in: memref<4xi32>, %out: memref<4xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %zero = arith.constant 0 : i32
  %ten = arith.constant 10 : i32
  %t1 = memref.alloc() : memref<4xi32>
  scf.for %i = %c0 to %c4 step %c1 {
    %v = memref.load %in[%i] : memref<4xi32>
    memref.store %v, %t1[%i] : memref<4xi32>
  }
  %t2 = memref.alloc() : memref<4xi32>
  %first = memref.load %in[%c0] : memref<4xi32>
  %neg = arith.cmpi slt, %first, %zero : i32
  cf.cond_br %neg, ^skip, ^fill
^fill:
  scf.for %i = %c0 to %c4 step %c1 {
    memref.store %ten, %t2[%i] : memref<4xi32>
  }
  scf.for %i = %c0 to %c4 step %c1 {
    %a = memref.load %t1[%i] : memref<4xi32>
    %b = memref.load %t2[%i] : memref<4xi32>
    %s = arith.addi %a, %b : i32
    memref.store %s, %out[%i] : memref<4xi32>
  }
  return
^skip:
  return
}
