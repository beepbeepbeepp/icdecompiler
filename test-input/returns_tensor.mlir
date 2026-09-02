module {
  func.func @add_two(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> {
    %0 = tosa.add %a, %b : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}