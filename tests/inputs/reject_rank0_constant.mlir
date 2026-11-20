func.func @f(%a: tensor<4xi32>) -> tensor<i32> {
  %c = arith.constant dense<7> : tensor<i32>
  return %c : tensor<i32>
}
