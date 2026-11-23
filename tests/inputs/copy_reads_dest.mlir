// Reverses a buffer through a temporary: tmp[i] = buf[3 - i], then tmp is
// copied back into buf. Writing straight into buf instead of tmp would
// overwrite elements that are still to be read.

func.func @reverse(%buf: memref<4xi32>){
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %tmp = memref.alloc() : memref<4xi32>
    scf.for %i = %c0 to %c4 step %c1 {
        %j = arith.subi %c3, %i : index
        %v = memref.load %buf[%j] : memref<4xi32>
        memref.store %v, %tmp[%i] : memref<4xi32>
    }
    memref.copy %tmp, %buf : memref<4xi32> to memref<4xi32>
    return
}