func.func @f(%a: tensor<?x4xi16>, %b: tensor<?x4xi16>) -> tensor<?x4xi16> {
  %0 = tosa.add %a, %b : (tensor<?x4xi16>, tensor<?x4xi16>) -> tensor<?x4xi16>
  return %0 : tensor<?x4xi16>
}
