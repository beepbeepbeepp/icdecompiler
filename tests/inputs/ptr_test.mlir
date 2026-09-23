emitc.include <"systolic_drv.h">

emitc.func @test_ptr(%A: !emitc.array<8x12xf32>) {
  %c0 = "emitc.constant"() <{value = 0 : index}> : () -> index
  %elem = emitc.subscript %A[%c0, %c0] : (!emitc.array<8x12xf32>, index, index) -> !emitc.lvalue<f32>
  %ptr = emitc.address_of %elem : !emitc.lvalue<f32>
  emitc.call_opaque "dummy_use"(%ptr) : (!emitc.ptr<f32>) -> ()
  emitc.return
}