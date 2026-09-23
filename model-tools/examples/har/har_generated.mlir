module {
  func.func @har_model(%p1: tensor<1x512x3xi8>, %p4: tensor<16x1x15x3xi8>, %p5: tensor<16xi32>, %p24: tensor<32x1x7x16xi8>, %p25: tensor<32xi32>, %p44: tensor<64x1x3x32xi8>, %p45: tensor<64xi32>, %p70: tensor<1x64x32xi8>, %p76: tensor<32xi32>, %p88: tensor<1x32x4xi8>, %p94: tensor<4xi32>, %dst103: memref<1x4xi8>) {
    %s2 = tosa.const_shape {values = dense<[1, 1, 512, 3]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %v3 = tosa.reshape %p1, %s2 : (tensor<1x512x3xi8>, !tosa.shape<4>) -> tensor<1x1x512x3xi8>
    %c6 = "tosa.const"() {values = dense<-8> : tensor<1xi8>} : () -> tensor<1xi8>
    %c7 = "tosa.const"() {values = dense<0> : tensor<1xi8>} : () -> tensor<1xi8>
    %a8 = tosa.conv2d %v3, %p4, %p5, %c6, %c7 { acc_type = i32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 6, 7>, stride = array<i64: 1, 2> } : (tensor<1x1x512x3xi8>, tensor<16x1x15x3xi8>, tensor<16xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x1x256x16xi32>
    %c9 = "tosa.const"() {values = dense<[1112281741, 2074679821, 2091616258, 1582152483, 1911933982, 1521272147, 1110744069, 1699089266, 1910484989, 1890344131, 1963073269, 1499455161, 1117216755, 1704859301, 1901998517, 1390421503]> : tensor<16xi32>} : () -> tensor<16xi32>
    %c10 = "tosa.const"() {values = dense<[38, 39, 39, 39, 39, 39, 38, 39, 39, 39, 39, 39, 38, 39, 39, 39]> : tensor<16xi8>} : () -> tensor<16xi8>
    %c11 = "tosa.const"() {values = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
    %c12 = "tosa.const"() {values = dense<-128> : tensor<1xi8>} : () -> tensor<1xi8>
    %r13 = tosa.rescale %a8, %c9, %c10, %c11, %c12 { scale32 = true, rounding_mode = #tosa.rounding_mode<DOUBLE_ROUND>, per_channel = true, input_unsigned = false, output_unsigned = false } : (tensor<1x1x256x16xi32>, tensor<16xi32>, tensor<16xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x1x256x16xi8>
    %v14 = tosa.clamp %r13 { min_val = -128 : i8, max_val = 127 : i8 } : (tensor<1x1x256x16xi8>) -> tensor<1x1x256x16xi8>
    %s15 = tosa.const_shape {values = dense<[1, 256, 16]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %v16 = tosa.reshape %v14, %s15 : (tensor<1x1x256x16xi8>, !tosa.shape<3>) -> tensor<1x256x16xi8>
    %s17 = tosa.const_shape {values = dense<[1, 1, 256, 16]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %v18 = tosa.reshape %v16, %s17 : (tensor<1x256x16xi8>, !tosa.shape<4>) -> tensor<1x1x256x16xi8>
    %v19 = tosa.max_pool2d %v18 { kernel = array<i64: 1, 2>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 2> } : (tensor<1x1x256x16xi8>) -> tensor<1x1x128x16xi8>
    %s20 = tosa.const_shape {values = dense<[1, 128, 16]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %v21 = tosa.reshape %v19, %s20 : (tensor<1x1x128x16xi8>, !tosa.shape<3>) -> tensor<1x128x16xi8>
    %s22 = tosa.const_shape {values = dense<[1, 1, 128, 16]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %v23 = tosa.reshape %v21, %s22 : (tensor<1x128x16xi8>, !tosa.shape<4>) -> tensor<1x1x128x16xi8>
    %c26 = "tosa.const"() {values = dense<-128> : tensor<1xi8>} : () -> tensor<1xi8>
    %c27 = "tosa.const"() {values = dense<0> : tensor<1xi8>} : () -> tensor<1xi8>
    %a28 = tosa.conv2d %v23, %p24, %p25, %c26, %c27 { acc_type = i32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 2, 3>, stride = array<i64: 1, 2> } : (tensor<1x1x128x16xi8>, tensor<32x1x7x16xi8>, tensor<32xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x1x64x32xi32>
    %c29 = "tosa.const"() {values = dense<[1698087857, 1670510023, 1869491191, 1087388958, 1076496903, 1380294647, 1721849973, 1260775166, 1765895417, 1215056156, 2064099354, 1380977263, 1495101518, 1811592281, 1907348794, 1641395389, 2127433767, 2049666575, 1129069092, 1953631784, 1456219625, 2035338197, 1554738485, 2049185215, 1603152674, 1340838432, 1221530793, 2051702744, 2014697500, 1909175189, 1904852315, 1715078806]> : tensor<32xi32>} : () -> tensor<32xi32>
    %c30 = "tosa.const"() {values = dense<[39, 39, 39, 38, 38, 38, 39, 38, 39, 38, 39, 38, 40, 39, 39, 39, 39, 39, 38, 39, 38, 39, 38, 39, 38, 38, 38, 39, 39, 39, 39, 39]> : tensor<32xi8>} : () -> tensor<32xi8>
    %c31 = "tosa.const"() {values = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
    %c32 = "tosa.const"() {values = dense<-128> : tensor<1xi8>} : () -> tensor<1xi8>
    %r33 = tosa.rescale %a28, %c29, %c30, %c31, %c32 { scale32 = true, rounding_mode = #tosa.rounding_mode<DOUBLE_ROUND>, per_channel = true, input_unsigned = false, output_unsigned = false } : (tensor<1x1x64x32xi32>, tensor<32xi32>, tensor<32xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x1x64x32xi8>
    %v34 = tosa.clamp %r33 { min_val = -128 : i8, max_val = 127 : i8 } : (tensor<1x1x64x32xi8>) -> tensor<1x1x64x32xi8>
    %s35 = tosa.const_shape {values = dense<[1, 64, 32]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %v36 = tosa.reshape %v34, %s35 : (tensor<1x1x64x32xi8>, !tosa.shape<3>) -> tensor<1x64x32xi8>
    %s37 = tosa.const_shape {values = dense<[1, 1, 64, 32]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %v38 = tosa.reshape %v36, %s37 : (tensor<1x64x32xi8>, !tosa.shape<4>) -> tensor<1x1x64x32xi8>
    %v39 = tosa.max_pool2d %v38 { kernel = array<i64: 1, 2>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 2> } : (tensor<1x1x64x32xi8>) -> tensor<1x1x32x32xi8>
    %s40 = tosa.const_shape {values = dense<[1, 32, 32]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %v41 = tosa.reshape %v39, %s40 : (tensor<1x1x32x32xi8>, !tosa.shape<3>) -> tensor<1x32x32xi8>
    %s42 = tosa.const_shape {values = dense<[1, 1, 32, 32]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %v43 = tosa.reshape %v41, %s42 : (tensor<1x32x32xi8>, !tosa.shape<4>) -> tensor<1x1x32x32xi8>
    %c46 = "tosa.const"() {values = dense<-128> : tensor<1xi8>} : () -> tensor<1xi8>
    %c47 = "tosa.const"() {values = dense<0> : tensor<1xi8>} : () -> tensor<1xi8>
    %a48 = tosa.conv2d %v43, %p44, %p45, %c46, %c47 { acc_type = i32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 1, 1>, stride = array<i64: 1, 1> } : (tensor<1x1x32x32xi8>, tensor<64x1x3x32xi8>, tensor<64xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x1x32x64xi32>
    %c49 = "tosa.const"() {values = dense<[1556488641, 1282596200, 1450541093, 1839125922, 1132458583, 1141143291, 1358962854, 1259954155, 1329484621, 1984581004, 2109714079, 1101164534, 1120409239, 1399839619, 1478779009, 1571816237, 2078544840, 2028769093, 2036470583, 1421138832, 1241187054, 1107925117, 1343337242, 1775601832, 1301998199, 1946558272, 1744824472, 1819265313, 1078035573, 1101035912, 1228818868, 1490346056, 2043726222, 1593940714, 1878069249, 1707291149, 2038178016, 1116291057, 1598529899, 1698724943, 1112251045, 2068364770, 1645447213, 1168555796, 1897976790, 1662065136, 2038601722, 1815103205, 1246585371, 1185664596, 1810687667, 1075511895, 1155512527, 2016206801, 1115519324, 1093044913, 2012935369, 1448379449, 1538007530, 2144533284, 1084615909, 1629241408, 1798743846, 1532715102]> : tensor<64xi32>} : () -> tensor<64xi32>
    %c50 = "tosa.const"() {values = dense<[39, 38, 38, 39, 39, 38, 39, 38, 39, 39, 39, 38, 38, 39, 39, 39, 39, 39, 39, 39, 38, 38, 38, 39, 39, 39, 39, 39, 39, 39, 38, 39, 39, 39, 39, 39, 39, 39, 39, 39, 38, 39, 39, 38, 39, 39, 39, 39, 38, 38, 39, 38, 39, 39, 38, 39, 39, 39, 39, 39, 38, 38, 40, 38]> : tensor<64xi8>} : () -> tensor<64xi8>
    %c51 = "tosa.const"() {values = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
    %c52 = "tosa.const"() {values = dense<-128> : tensor<1xi8>} : () -> tensor<1xi8>
    %r53 = tosa.rescale %a48, %c49, %c50, %c51, %c52 { scale32 = true, rounding_mode = #tosa.rounding_mode<DOUBLE_ROUND>, per_channel = true, input_unsigned = false, output_unsigned = false } : (tensor<1x1x32x64xi32>, tensor<64xi32>, tensor<64xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x1x32x64xi8>
    %v54 = tosa.clamp %r53 { min_val = -128 : i8, max_val = 127 : i8 } : (tensor<1x1x32x64xi8>) -> tensor<1x1x32x64xi8>
    %s55 = tosa.const_shape {values = dense<[1, 32, 64]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %v56 = tosa.reshape %v54, %s55 : (tensor<1x1x32x64xi8>, !tosa.shape<3>) -> tensor<1x32x64xi8>
    %v57 = tosa.cast %v56 : (tensor<1x32x64xi8>) -> tensor<1x32x64xi32>
    %a58 = tosa.reduce_sum %v57 {axis = 1 : i32} : (tensor<1x32x64xi32>) -> tensor<1x1x64xi32>
    %c59 = "tosa.const"() {values = dense<-4096> : tensor<1x1x1xi32>} : () -> tensor<1x1x1xi32>
    %a60 = tosa.sub %a58, %c59 : (tensor<1x1x64xi32>, tensor<1x1x1xi32>) -> tensor<1x1x64xi32>
    %c61 = "tosa.const"() {values = dense<1415597775> : tensor<1xi32>} : () -> tensor<1xi32>
    %c62 = "tosa.const"() {values = dense<33> : tensor<1xi8>} : () -> tensor<1xi8>
    %c63 = "tosa.const"() {values = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
    %c64 = "tosa.const"() {values = dense<-128> : tensor<1xi8>} : () -> tensor<1xi8>
    %r65 = tosa.rescale %a60, %c61, %c62, %c63, %c64 { scale32 = true, rounding_mode = #tosa.rounding_mode<DOUBLE_ROUND>, per_channel = false, input_unsigned = false, output_unsigned = false } : (tensor<1x1x64xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x1x64xi8>
    %s66 = tosa.const_shape {values = dense<[1, 64]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %v67 = tosa.reshape %r65, %s66 : (tensor<1x1x64xi8>, !tosa.shape<2>) -> tensor<1x64xi8>
    %s68 = tosa.const_shape {values = dense<[1, 1, 64]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %v69 = tosa.reshape %v67, %s68 : (tensor<1x64xi8>, !tosa.shape<3>) -> tensor<1x1x64xi8>
    %c71 = "tosa.const"() {values = dense<-128> : tensor<1xi8>} : () -> tensor<1xi8>
    %c72 = "tosa.const"() {values = dense<0> : tensor<1xi8>} : () -> tensor<1xi8>
    %a73 = tosa.matmul %v69, %p70, %c71, %c72 : (tensor<1x1x64xi8>, tensor<1x64x32xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x1x32xi32>
    %s74 = tosa.const_shape {values = dense<[1, 1, 32]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %v75 = tosa.reshape %p76, %s74 : (tensor<32xi32>, !tosa.shape<3>) -> tensor<1x1x32xi32>
    %a77 = tosa.add %a73, %v75 : (tensor<1x1x32xi32>, tensor<1x1x32xi32>) -> tensor<1x1x32xi32>
    %s78 = tosa.const_shape {values = dense<[1, 32]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %v79 = tosa.reshape %a77, %s78 : (tensor<1x1x32xi32>, !tosa.shape<2>) -> tensor<1x32xi32>
    %c80 = "tosa.const"() {values = dense<[1340263547, 1601737968, 1764798188, 1610401602, 1555765026, 1462560932, 1210314882, 1591183099, 1546101986, 1219307829, 1620981182, 1085883588, 1813578758, 1358738489, 1314303128, 1701608069, 2059215696, 1322704407, 1248663474, 1303295611, 1447981511, 1807709286, 1606563791, 1392998827, 1316417061, 1774344921, 1211899962, 1499212355, 1449422613, 1216989619, 1438447947, 1121743951]> : tensor<32xi32>} : () -> tensor<32xi32>
    %c81 = "tosa.const"() {values = dense<[40, 40, 40, 41, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 41, 40, 40, 40, 40, 41, 41, 40, 40, 40, 40, 40, 40, 40, 41, 40]> : tensor<32xi8>} : () -> tensor<32xi8>
    %c82 = "tosa.const"() {values = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
    %c83 = "tosa.const"() {values = dense<-128> : tensor<1xi8>} : () -> tensor<1xi8>
    %r84 = tosa.rescale %v79, %c80, %c81, %c82, %c83 { scale32 = true, rounding_mode = #tosa.rounding_mode<DOUBLE_ROUND>, per_channel = true, input_unsigned = false, output_unsigned = false } : (tensor<1x32xi32>, tensor<32xi32>, tensor<32xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x32xi8>
    %v85 = tosa.clamp %r84 { min_val = -128 : i8, max_val = 127 : i8 } : (tensor<1x32xi8>) -> tensor<1x32xi8>
    %s86 = tosa.const_shape {values = dense<[1, 1, 32]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %v87 = tosa.reshape %v85, %s86 : (tensor<1x32xi8>, !tosa.shape<3>) -> tensor<1x1x32xi8>
    %c89 = "tosa.const"() {values = dense<-128> : tensor<1xi8>} : () -> tensor<1xi8>
    %c90 = "tosa.const"() {values = dense<0> : tensor<1xi8>} : () -> tensor<1xi8>
    %a91 = tosa.matmul %v87, %p88, %c89, %c90 : (tensor<1x1x32xi8>, tensor<1x32x4xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x1x4xi32>
    %s92 = tosa.const_shape {values = dense<[1, 1, 4]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %v93 = tosa.reshape %p94, %s92 : (tensor<4xi32>, !tosa.shape<3>) -> tensor<1x1x4xi32>
    %a95 = tosa.add %a91, %v93 : (tensor<1x1x4xi32>, tensor<1x1x4xi32>) -> tensor<1x1x4xi32>
    %s96 = tosa.const_shape {values = dense<[1, 4]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %v97 = tosa.reshape %a95, %s96 : (tensor<1x1x4xi32>, !tosa.shape<2>) -> tensor<1x4xi32>
    %c98 = "tosa.const"() {values = dense<[1147040630, 1288448682, 2145581135, 1698747304]> : tensor<4xi32>} : () -> tensor<4xi32>
    %c99 = "tosa.const"() {values = dense<[39, 39, 40, 40]> : tensor<4xi8>} : () -> tensor<4xi8>
    %c100 = "tosa.const"() {values = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
    %c101 = "tosa.const"() {values = dense<-16> : tensor<1xi8>} : () -> tensor<1xi8>
    %r102 = tosa.rescale %v97, %c98, %c99, %c100, %c101 { scale32 = true, rounding_mode = #tosa.rounding_mode<DOUBLE_ROUND>, per_channel = true, input_unsigned = false, output_unsigned = false } : (tensor<1x4xi32>, tensor<4xi32>, tensor<4xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x4xi8>
    bufferization.materialize_in_destination %r102 in restrict writable %dst103 : (tensor<1x4xi8>, memref<1x4xi8>) -> ()
    return
  }
}
