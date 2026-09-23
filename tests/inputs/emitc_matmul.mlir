emitc.include <"systolic_drv.h">

emitc.func @run_matmul(%M: i32, %K: i32, %N: i32,
                       %A: !emitc.ptr<i16>, %B: !emitc.ptr<i16>,
                       %Res: !emitc.ptr<i32>) {
  emitc.call_opaque "systolic_multiply_large"(%M, %K, %N, %A, %B, %Res)
    : (i32, i32, i32, !emitc.ptr<i16>, !emitc.ptr<i16>, !emitc.ptr<i32>) -> ()
  emitc.return
}