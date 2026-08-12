emitc.include <"systolic_drv.h">

func.func @run_big(%A: memref<16x32xf32>, %B: memref<32x16xf32>, %C: memref<16x16xf32>) {
  linalg.matmul ins(%A, %B: memref<16x32xf32>, memref<32x16xf32>)
                outs(%C: memref<16x16xf32>)
  return
}