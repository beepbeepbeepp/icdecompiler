emitc.include <"systolic_drv.h">

func.func @run_multi(%A: memref<8x8xf32>, %B: memref<8x8xf32>, %C: memref<8x8xf32>,
                     %D: memref<8x8xf32>, %E: memref<8x8xf32>) {
  linalg.matmul ins(%A, %B: memref<8x8xf32>, memref<8x8xf32>)
                outs(%C: memref<8x8xf32>)
  linalg.matmul ins(%C, %D: memref<8x8xf32>, memref<8x8xf32>)
                outs(%E: memref<8x8xf32>)
  return
}