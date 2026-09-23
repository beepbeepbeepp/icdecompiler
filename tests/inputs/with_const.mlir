module {
  func.func @scale(%a: tensor<4xf32>) -> tensor<4xf32> {
    %c = "tosa.const"() {values = dense<0.5> : tensor<4xf32>} : () -> tensor<4xf32>
    %sh = "tosa.const"() {values = dense<0> : tensor<1xi8>} : () -> tensor<1xi8>
    %0 = tosa.mul %a, %c, %sh : (tensor<4xf32>, tensor<4xf32>, tensor<1xi8>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}