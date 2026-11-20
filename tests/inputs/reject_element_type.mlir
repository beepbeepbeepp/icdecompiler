func.func @f(%a: tensor<4xui8>, %b: tensor<4xui8>) -> tensor<4xui8> {
  %0 = tosa.add %a, %b : (tensor<4xui8>, tensor<4xui8>) -> tensor<4xui8>
  return %0 : tensor<4xui8>
}
