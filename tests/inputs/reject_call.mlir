func.func @g(%a: tensor<4xi16>) -> tensor<4xi16> {
  %0 = tosa.add %a, %a : (tensor<4xi16>, tensor<4xi16>) -> tensor<4xi16>
  return %0 : tensor<4xi16>
}
func.func @f(%a: tensor<4xi16>) -> tensor<4xi16> {
  %0 = call @g(%a) : (tensor<4xi16>) -> tensor<4xi16>
  return %0 : tensor<4xi16>
}
