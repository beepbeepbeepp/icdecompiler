emitc.include <"systolic_drv.h">

func.func @run_matmul(%A: memref<8x8xf32>, %B: memref<8x8xf32>, %C: memref<8x8xf32>) {
  linalg.matmul ins(%A, %B: memref<8x8xf32>, memref<8x8xf32>)
                outs(%C: memref<8x8xf32>)
  return
}