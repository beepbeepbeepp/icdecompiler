#!/usr/bin/env python3
# Copyright 2026 Indonesia Chip Design Collaborative Center (ICDeC)
# Author: Muhammad Faudzan Abdullah
# SPDX-License-Identifier: Apache-2.0

"""
Generate TOSA MLIR from a quantized TFLite model.

Covers the operator set used by the lab's biomedical 1D-CNN models. Weights are
emitted as function parameters, not constants, so the MLIR stays small and the
actual values come from extract_weights.py.

Adding an operator means writing one function and registering it in HANDLERS.
Nothing else needs to change.

Usage:
    python3 tflite_to_tosa.py model_int8.tflite -o har_model.mlir --name har_model
"""

import argparse
import math

import numpy as np
import tflite


# ---------------------------------------------------------------- helpers

def quantize_multiplier(m):
    """TFLite QuantizeMultiplier: m -> (int32 multiplier, right shift)."""
    if m == 0:
        return 0, 0
    frac, exp = math.frexp(m)
    fixed = int(round(frac * (1 << 31)))
    if fixed == (1 << 31):
        fixed //= 2
        exp += 1
    return fixed, 31 - exp


def mlir_type(shape, dtype):
    return "tensor<" + "x".join(str(d) for d in shape) + f"x{dtype}>"


DTYPE = {0: "f32", 2: "i32", 9: "i8"}


# ---------------------------------------------------------------- model

class TFLiteModel:
    def __init__(self, path):
        with open(path, "rb") as f:
            self.m = tflite.Model.GetRootAsModel(f.read(), 0)
        self.sg = self.m.Subgraphs(0)
        table = {getattr(tflite.BuiltinOperator, n): n
                 for n in dir(tflite.BuiltinOperator) if not n.startswith("_")}
        self.opnames = [table.get(self.m.OperatorCodes(i).BuiltinCode(), "?")
                        for i in range(self.m.OperatorCodesLength())]

    def shape(self, idx):
        t = self.sg.Tensors(idx)
        return [t.Shape(j) for j in range(t.ShapeLength())]

    def dtype(self, idx):
        return DTYPE[self.sg.Tensors(idx).Type()]

    def quant(self, idx):
        q = self.sg.Tensors(idx).Quantization()
        scales = np.array([q.Scale(i) for i in range(q.ScaleLength())], dtype=np.float64)
        zps = np.array([q.ZeroPoint(i) for i in range(q.ZeroPointLength())])
        return scales, zps

    def is_constant(self, idx):
        return self.m.Buffers(self.sg.Tensors(idx).Buffer()).DataLength() > 0

    def operators(self):
        for i in range(self.sg.OperatorsLength()):
            op = self.sg.Operators(i)
            yield Op(kind=self.opnames[op.OpcodeIndex()],
                     inputs=[op.Inputs(j) for j in range(op.InputsLength())],
                     outputs=[op.Outputs(j) for j in range(op.OutputsLength())],
                     raw=op)


class Op:
    def __init__(self, kind, inputs, outputs, raw):
        self.kind = kind
        self.inputs = inputs
        self.outputs = outputs
        self.raw = raw


# ---------------------------------------------------------------- builder

class Builder:
    """Collects MLIR lines and hands out SSA names."""

    def __init__(self, model):
        self.model = model
        self.lines = []
        self.names = {}          # tensor index -> SSA name
        self.params = []         # (ssa name, mlir type) for the function signature
        self._counter = 0

    def fresh(self, hint="v"):
        self._counter += 1
        return f"%{hint}{self._counter}"

    def emit(self, line):
        self.lines.append("    " + line)

    def name_of(self, idx):
        """SSA name for a tensor, creating a function parameter if it is a constant."""
        if idx in self.names:
            return self.names[idx]
        m = self.model
        n = self.fresh("p")
        self.params.append((n, mlir_type(m.shape(idx), m.dtype(idx))))
        self.names[idx] = n
        return n

    def name_of_as(self, idx, shape):
        """Like name_of, but declares the parameter with an overridden shape."""
        if idx in self.names:
            return self.names[idx]
        n = self.fresh("p")
        self.params.append((n, mlir_type(shape, self.model.dtype(idx))))
        self.names[idx] = n
        return n

    def bind(self, idx, ssa):
        self.names[idx] = ssa

    def const(self, values, shape, dtype):
        """Emit a tosa.const and return its name."""
        n = self.fresh("c")
        if len(values) == 1:
            body = f"dense<{values[0]}>"
        else:
            body = "dense<[" + ", ".join(str(v) for v in values) + "]>"
        ty = mlir_type(shape, dtype)
        self.emit(f'{n} = "tosa.const"() {{values = {body} : {ty}}} : () -> {ty}')
        return n

    def rescale(self, src, src_idx, dst_idx, weight_idx=None):
        """Emit tosa.rescale from an i32 accumulator to the output tensor type."""
        m = self.model
        in_scales, in_zps = m.quant(src_idx)
        out_scales, out_zps = m.quant(dst_idx)
        if weight_idx is not None:
            w_scales, _ = m.quant(weight_idx)
            mult = (in_scales[0] * w_scales) / out_scales[0]
        else:
            mult = np.array([in_scales[0] / out_scales[0]])

        pairs = [quantize_multiplier(v) for v in mult]
        m0 = [p[0] for p in pairs]
        shifts = [p[1] for p in pairs]

        n_mul = self.const(m0, [len(m0)], "i32")
        n_shf = self.const(shifts, [len(shifts)], "i8")
        n_izp = self.const([0], [1], "i32")
        n_ozp = self.const([int(out_zps[0])], [1], "i8")

        acc_ty = mlir_type(m.shape(dst_idx), "i32")
        out_ty = mlir_type(m.shape(dst_idx), "i8")
        n = self.fresh("r")
        self.emit(
            f"{n} = tosa.rescale {src}, {n_mul}, {n_shf}, {n_izp}, {n_ozp} {{"
            f" scale32 = true,"
            f" rounding_mode = #tosa.rounding_mode<DOUBLE_ROUND>,"
            f" per_channel = {'true' if len(m0) > 1 else 'false'},"
            f" input_unsigned = false, output_unsigned = false }}"
            f" : ({acc_ty}, tensor<{len(m0)}xi32>, tensor<{len(shifts)}xi8>,"
            f" tensor<1xi32>, tensor<1xi8>) -> {out_ty}")
        return n


# ---------------------------------------------------------------- handlers
# Each handler takes (builder, op) and binds the output tensor to an SSA value.
# Register new operators here; nothing else needs to change.

HANDLERS = {}


def handler(kind):
    def wrap(fn):
        HANDLERS[kind] = fn
        return fn
    return wrap

@handler("RESHAPE")
@handler("EXPAND_DIMS")
@handler("SQUEEZE")
def _reshape(b, op):
    """Shape-only ops all become tosa.reshape with the output shape."""
    src = b.name_of(op.inputs[0])
    dst = op.outputs[0]
    shape = b.model.shape(dst)
    dtype = b.model.dtype(dst)

    n_shape = b.fresh("s")
    b.emit(f"{n_shape} = tosa.const_shape {{values = dense<[{', '.join(map(str, shape))}]>"
           f" : tensor<{len(shape)}xindex>}} : () -> !tosa.shape<{len(shape)}>")

    n = b.fresh("v")
    src_ty = mlir_type(b.model.shape(op.inputs[0]), dtype)
    out_ty = mlir_type(shape, dtype)
    b.emit(f"{n} = tosa.reshape {src}, {n_shape} : ({src_ty}, !tosa.shape<{len(shape)}>) -> {out_ty}")
    b.bind(dst, n)


@handler("SOFTMAX")
def _softmax(b, op):
    """Skipped: classification only needs argmax over the logits."""
    b.bind(op.outputs[0], b.name_of(op.inputs[0]))

@handler("MAX_POOL_2D")
def _max_pool(b, op):
    """Pooling keeps the same quantization, so no rescale is needed."""
    o = op.raw.BuiltinOptions()
    opts = tflite.Pool2DOptions()
    opts.Init(o.Bytes, o.Pos)

    src = b.name_of(op.inputs[0])
    dst = op.outputs[0]
    dtype = b.model.dtype(dst)
    src_ty = mlir_type(b.model.shape(op.inputs[0]), dtype)
    out_ty = mlir_type(b.model.shape(dst), dtype)

    kh, kw = opts.FilterHeight(), opts.FilterWidth()
    sh, sw = opts.StrideH(), opts.StrideW()

    n = b.fresh("v")
    b.emit(f"{n} = tosa.max_pool2d {src} {{"
           f" kernel = array<i64: {kh}, {kw}>,"
           f" pad = array<i64: 0, 0, 0, 0>,"
           f" stride = array<i64: {sh}, {sw}> }}"
           f" : ({src_ty}) -> {out_ty}")
    b.bind(dst, n)

def _same_padding(in_size, k, stride, dilation=1):
    """TFLite SAME padding, split as TOSA expects (before, after)."""
    eff_k = (k - 1) * dilation + 1
    out = (in_size + stride - 1) // stride
    total = max(0, (out - 1) * stride + eff_k - in_size)
    return total // 2, total - total // 2


@handler("CONV_2D")
def _conv2d(b, op):
    o = op.raw.BuiltinOptions()
    opts = tflite.Conv2DOptions()
    opts.Init(o.Bytes, o.Pos)

    x_idx, w_idx, bias_idx = op.inputs[0], op.inputs[1], op.inputs[2]
    dst = op.outputs[0]
    m = b.model

    x = b.name_of(x_idx)
    w = b.name_of(w_idx)
    bias = b.name_of(bias_idx)

    in_shape = m.shape(x_idx)          # [N, H, W, C]
    w_shape = m.shape(w_idx)           # [F, KH, KW, C]
    out_shape = m.shape(dst)

    sh, sw = opts.StrideH(), opts.StrideW()
    dh, dw = opts.DilationHFactor(), opts.DilationWFactor()

    # padding: 1 = SAME, 2 = VALID
    if opts.Padding() == tflite.Padding.SAME:
        pt, pb = _same_padding(in_shape[1], w_shape[1], sh, dh)
        pl, pr = _same_padding(in_shape[2], w_shape[2], sw, dw)
    else:
        pt = pb = pl = pr = 0

    _, in_zps = m.quant(x_idx)
    _, w_zps = m.quant(w_idx)

    n_izp = b.const([int(in_zps[0])], [1], "i8")
    n_wzp = b.const([int(w_zps[0])], [1], "i8")

    acc_ty = mlir_type(out_shape, "i32")
    n_acc = b.fresh("a")
    b.emit(f"{n_acc} = tosa.conv2d {x}, {w}, {bias}, {n_izp}, {n_wzp} {{"
           f" acc_type = i32,"
           f" dilation = array<i64: {dh}, {dw}>,"
           f" pad = array<i64: {pt}, {pb}, {pl}, {pr}>,"
           f" stride = array<i64: {sh}, {sw}> }}"
           f" : ({mlir_type(in_shape, 'i8')}, {mlir_type(w_shape, 'i8')},"
           f" {mlir_type(m.shape(bias_idx), 'i32')},"
           f" tensor<1xi8>, tensor<1xi8>) -> {acc_ty}")

    n_out = b.rescale(n_acc, x_idx, dst, weight_idx=w_idx)

    # fused relu clamps at the output zero point
    if opts.FusedActivationFunction() == 1:      # RELU
        _, out_zps = m.quant(dst)
        zp = int(out_zps[0])
        n_clamp = b.fresh("v")
        out_ty = mlir_type(out_shape, "i8")
        b.emit(f"{n_clamp} = tosa.clamp {n_out} {{"
               f" min_val = {zp} : i8, max_val = 127 : i8 }}"
               f" : ({out_ty}) -> {out_ty}")
        n_out = n_clamp

    b.bind(dst, n_out)

@handler("FULLY_CONNECTED")
def _fully_connected(b, op):
    """TFLite stores weights as [out, in]; TOSA matmul wants [1, in, out]."""
    o = op.raw.BuiltinOptions()
    opts = tflite.FullyConnectedOptions()
    opts.Init(o.Bytes, o.Pos)

    x_idx, w_idx, bias_idx = op.inputs[0], op.inputs[1], op.inputs[2]
    dst = op.outputs[0]
    m = b.model

    in_shape = m.shape(x_idx)          # [N, C_in]
    w_shape = m.shape(w_idx)           # [C_out, C_in]
    out_shape = m.shape(dst)           # [N, C_out]
    n_in, n_out = w_shape[1], w_shape[0]

    # matmul needs a batch dimension on both operands
    x3 = b.fresh("s")
    b.emit(f"{x3} = tosa.const_shape {{values = dense<[1, {in_shape[0]}, {n_in}]>"
           f" : tensor<3xindex>}} : () -> !tosa.shape<3>")
    x_r = b.fresh("v")
    b.emit(f"{x_r} = tosa.reshape {b.name_of(x_idx)}, {x3}"
           f" : ({mlir_type(in_shape, 'i8')}, !tosa.shape<3>)"
           f" -> {mlir_type([1, in_shape[0], n_in], 'i8')}")

    # the weight parameter is declared already transposed to [1, in, out]
    w = b.name_of_as(w_idx, [1, n_in, n_out])

    _, in_zps = m.quant(x_idx)
    _, w_zps = m.quant(w_idx)
    n_azp = b.const([int(in_zps[0])], [1], "i8")
    n_bzp = b.const([int(w_zps[0])], [1], "i8")

    acc3 = [1, in_shape[0], n_out]
    n_mm = b.fresh("a")
    b.emit(f"{n_mm} = tosa.matmul {x_r}, {w}, {n_azp}, {n_bzp}"
           f" : ({mlir_type([1, in_shape[0], n_in], 'i8')},"
           f" {mlir_type([1, n_in, n_out], 'i8')},"
           f" tensor<1xi8>, tensor<1xi8>) -> {mlir_type(acc3, 'i32')}")

    # add bias, broadcast over the batch dimension. TFLite drops an all-zero
    # bias and marks the slot -1, so it may simply not be there.
    if bias_idx < 0:
        n_add = n_mm
    else:
        bs = b.fresh("s")
        b.emit(f"{bs} = tosa.const_shape {{values = dense<[1, 1, {n_out}]>"
               f" : tensor<3xindex>}} : () -> !tosa.shape<3>")
        b_r = b.fresh("v")
        b.emit(f"{b_r} = tosa.reshape {b.name_of(bias_idx)}, {bs}"
               f" : ({mlir_type(m.shape(bias_idx), 'i32')}, !tosa.shape<3>)"
               f" -> {mlir_type([1, 1, n_out], 'i32')}")
        n_add = b.fresh("a")
        b.emit(f"{n_add} = tosa.add {n_mm}, {b_r}"
               f" : ({mlir_type(acc3, 'i32')}, {mlir_type([1, 1, n_out], 'i32')})"
               f" -> {mlir_type(acc3, 'i32')}")

    # back to the 2-D output shape before rescaling
    os_ = b.fresh("s")
    b.emit(f"{os_} = tosa.const_shape {{values = dense<[{', '.join(map(str, out_shape))}]>"
           f" : tensor<{len(out_shape)}xindex>}} : () -> !tosa.shape<{len(out_shape)}>")
    n_flat = b.fresh("v")
    b.emit(f"{n_flat} = tosa.reshape {n_add}, {os_}"
           f" : ({mlir_type(acc3, 'i32')}, !tosa.shape<{len(out_shape)}>)"
           f" -> {mlir_type(out_shape, 'i32')}")

    n_out_v = b.rescale(n_flat, x_idx, dst, weight_idx=w_idx)

    if opts.FusedActivationFunction() == 1:      # RELU
        _, out_zps = m.quant(dst)
        zp = int(out_zps[0])
        ty = mlir_type(out_shape, "i8")
        n_clamp = b.fresh("v")
        b.emit(f"{n_clamp} = tosa.clamp {n_out_v} {{"
               f" min_val = {zp} : i8, max_val = 127 : i8 }} : ({ty}) -> {ty}")
        n_out_v = n_clamp

    b.bind(dst, n_out_v)


@handler("MEAN")
def _mean(b, op):
    """Mean over one axis becomes reduce_sum plus a rescale that folds in 1/N."""
    m = b.model
    x_idx = op.inputs[0]
    dst = op.outputs[0]

    in_shape = m.shape(x_idx)
    out_shape = m.shape(dst)

    # the reduced axis is the one that disappears
    axis = next(i for i, d in enumerate(in_shape)
                if i >= len(out_shape) or out_shape[i] != d)
    count = in_shape[axis]

    # widen to 32 bits first: TOSA requires reduce_sum to keep the element type
    n_wide = b.fresh("v")
    b.emit(f"{n_wide} = tosa.cast {b.name_of(x_idx)}"
           f" : ({mlir_type(in_shape, 'i8')}) -> {mlir_type(in_shape, 'i32')}")

    sum_shape = list(in_shape)
    sum_shape[axis] = 1
    n_sum = b.fresh("a")
    b.emit(f"{n_sum} = tosa.reduce_sum {n_wide} {{axis = {axis} : i32}}"
           f" : ({mlir_type(in_shape, 'i32')}) -> {mlir_type(sum_shape, 'i32')}")

    # remove the accumulated input zero point before rescaling: TOSA requires
    # input_zp to be zero when the input is already 32-bit
    in_scales, in_zps = m.quant(x_idx)
    out_scales, out_zps = m.quant(dst)
    offset = int(in_zps[0]) * count
    if offset != 0:
        n_off = b.const([offset], [1, 1, 1][:len(sum_shape)], "i32")
        n_sub = b.fresh("a")
        b.emit(f"{n_sub} = tosa.sub {n_sum}, {n_off}"
               f" : ({mlir_type(sum_shape, 'i32')},"
               f" {mlir_type([1] * len(sum_shape), 'i32')})"
               f" -> {mlir_type(sum_shape, 'i32')}")
        n_sum = n_sub

    mult = in_scales[0] / count / out_scales[0]
    m0, shift = quantize_multiplier(mult)

    n_mul = b.const([m0], [1], "i32")
    n_shf = b.const([shift], [1], "i8")
    n_izp = b.const([0], [1], "i32")
    n_ozp = b.const([int(out_zps[0])], [1], "i8")

    n_rs = b.fresh("r")
    b.emit(f"{n_rs} = tosa.rescale {n_sum}, {n_mul}, {n_shf}, {n_izp}, {n_ozp} {{"
           f" scale32 = true,"
           f" rounding_mode = #tosa.rounding_mode<DOUBLE_ROUND>,"
           f" per_channel = false,"
           f" input_unsigned = false, output_unsigned = false }}"
           f" : ({mlir_type(sum_shape, 'i32')}, tensor<1xi32>, tensor<1xi8>,"
           f" tensor<1xi32>, tensor<1xi8>) -> {mlir_type(sum_shape, 'i8')}")

    # drop the reduced axis to match the output shape
    os_ = b.fresh("s")
    b.emit(f"{os_} = tosa.const_shape {{values = dense<[{', '.join(map(str, out_shape))}]>"
           f" : tensor<{len(out_shape)}xindex>}} : () -> !tosa.shape<{len(out_shape)}>")
    n = b.fresh("v")
    b.emit(f"{n} = tosa.reshape {n_rs}, {os_}"
           f" : ({mlir_type(sum_shape, 'i8')}, !tosa.shape<{len(out_shape)}>)"
           f" -> {mlir_type(out_shape, 'i8')}")
    b.bind(dst, n)

def _fused_relu(b, act, src, dst_idx):
    """Apply a fused activation, clamping at the output zero point."""
    if act == 0:                                   # NONE
        return src
    if act != 1:                                   # only RELU for now
        raise NotImplementedError(f"fused activation {act} not supported")
    _, out_zps = b.model.quant(dst_idx)
    zp = int(out_zps[0])
    ty = mlir_type(b.model.shape(dst_idx), "i8")
    n = b.fresh("v")
    b.emit(f"{n} = tosa.clamp {src} {{ min_val = {zp} : i8, max_val = 127 : i8 }}"
           f" : ({ty}) -> {ty}")
    return n


@handler("DEPTHWISE_CONV_2D")
def _depthwise_conv2d(b, op):
    """TFLite stores the filter as [1, KH, KW, C*M]; TOSA wants [KH, KW, C, M].
    The bytes are already in that order, so only the declared shape changes."""
    o = op.raw.BuiltinOptions()
    opts = tflite.DepthwiseConv2DOptions()
    opts.Init(o.Bytes, o.Pos)

    x_idx, w_idx = op.inputs[0], op.inputs[1]
    bias_idx = op.inputs[2] if len(op.inputs) > 2 else -1
    dst = op.outputs[0]
    m = b.model

    in_shape = m.shape(x_idx)                      # [N, H, W, C]
    w_tfl = m.shape(w_idx)                         # [1, KH, KW, C*M]
    out_shape = m.shape(dst)
    C = in_shape[3]
    KH, KW, CM = w_tfl[1], w_tfl[2], w_tfl[3]
    w_shape = [KH, KW, C, CM // C]

    x = b.name_of(x_idx)
    w = b.name_of_as(w_idx, w_shape)
    bias = b.name_of(bias_idx)

    sh, sw = opts.StrideH(), opts.StrideW()
    dh, dw = opts.DilationHFactor(), opts.DilationWFactor()
    if opts.Padding() == tflite.Padding.SAME:
        pt, pb = _same_padding(in_shape[1], KH, sh, dh)
        pl, pr = _same_padding(in_shape[2], KW, sw, dw)
    else:
        pt = pb = pl = pr = 0

    _, in_zps = m.quant(x_idx)
    _, w_zps = m.quant(w_idx)
    n_izp = b.const([int(in_zps[0])], [1], "i8")
    n_wzp = b.const([int(w_zps[0])], [1], "i8")

    n_acc = b.fresh("a")
    b.emit(f"{n_acc} = tosa.depthwise_conv2d {x}, {w}, {bias}, {n_izp}, {n_wzp} {{"
           f" acc_type = i32,"
           f" dilation = array<i64: {dh}, {dw}>,"
           f" pad = array<i64: {pt}, {pb}, {pl}, {pr}>,"
           f" stride = array<i64: {sh}, {sw}> }}"
           f" : ({mlir_type(in_shape, 'i8')}, {mlir_type(w_shape, 'i8')},"
           f" {mlir_type(m.shape(bias_idx), 'i32')},"
           f" tensor<1xi8>, tensor<1xi8>) -> {mlir_type(out_shape, 'i32')}")

    n_out = b.rescale(n_acc, x_idx, dst, weight_idx=w_idx)
    b.bind(dst, _fused_relu(b, opts.FusedActivationFunction(), n_out, dst))


@handler("ADD")
def _add(b, op):
    """Quantized add. The inputs usually have different scales, so each is
    brought to a shared i32 scale first, added, then rescaled to the output.
    Same left-shift-by-20 scheme as the TFLite reference kernel."""
    o = op.raw.BuiltinOptions()
    opts = tflite.AddOptions()
    opts.Init(o.Bytes, o.Pos)

    a_idx, c_idx = op.inputs[0], op.inputs[1]
    dst = op.outputs[0]
    m = b.model
    shape = list(m.shape(dst))
    if list(m.shape(a_idx)) != shape or list(m.shape(c_idx)) != shape:
        raise NotImplementedError("ADD with broadcasting is not supported yet")

    def q(idx):
        s, z = m.quant(idx)
        return float(s[0]), int(z[0])

    sa, za = q(a_idx)
    sc, zc = q(c_idx)
    so, zo = q(dst)

    left_shift = 20
    twice_max = 2.0 * max(sa, sc)
    ty8, ty32 = mlir_type(shape, "i8"), mlir_type(shape, "i32")

    def rescale(src, mult, in_zp, in_ty, in_zp_ty, out_zp, out_ty, out_zp_ty):
        m0, shift = quantize_multiplier(mult)
        n_mul = b.const([m0], [1], "i32")
        n_shf = b.const([shift], [1], "i8")
        n_izp = b.const([in_zp], [1], in_zp_ty)
        n_ozp = b.const([out_zp], [1], out_zp_ty)
        n = b.fresh("r")
        b.emit(f"{n} = tosa.rescale {src}, {n_mul}, {n_shf}, {n_izp}, {n_ozp} {{"
               f" scale32 = true,"
               f" rounding_mode = #tosa.rounding_mode<DOUBLE_ROUND>,"
               f" per_channel = false,"
               f" input_unsigned = false, output_unsigned = false }}"
               f" : ({in_ty}, tensor<1xi32>, tensor<1xi8>,"
               f" tensor<1x{in_zp_ty}>, tensor<1x{out_zp_ty}>) -> {out_ty}")
        return n

    ra = rescale(b.name_of(a_idx), sa / twice_max * (1 << left_shift),
                 za, ty8, "i8", 0, ty32, "i32")
    rc = rescale(b.name_of(c_idx), sc / twice_max * (1 << left_shift),
                 zc, ty8, "i8", 0, ty32, "i32")

    n_sum = b.fresh("a")
    b.emit(f"{n_sum} = tosa.add {ra}, {rc} : ({ty32}, {ty32}) -> {ty32}")

    n_out = rescale(n_sum, twice_max / ((1 << left_shift) * so),
                    0, ty32, "i32", zo, ty8, "i8")
    b.bind(dst, _fused_relu(b, opts.FusedActivationFunction(), n_out, dst))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--name", default="model", help="generated function name")
    args = ap.parse_args()

    m = TFLiteModel(args.model)

    unsupported = sorted({op.kind for op in m.operators() if op.kind not in HANDLERS})
    if unsupported:
        print("unsupported operators:", ", ".join(unsupported))
        print("add a handler for each in tflite_to_tosa.py")
        return 1

    b = Builder(m)

    # graph input becomes the first function parameter
    in_idx = m.sg.Inputs(0)
    b.name_of(in_idx)

    for op in m.operators():
        HANDLERS[op.kind](b, op)

    # the graph output is written into a destination buffer
    out_idx = m.sg.Outputs(0)
    out_shape = m.shape(out_idx)
    out_dtype = m.dtype(out_idx)
    dst = b.fresh("dst")
    result = b.names[out_idx]

    sig = ", ".join(f"{n}: {t}" for n, t in b.params)
    sig += f", {dst}: memref<" + "x".join(map(str, out_shape)) + f"x{out_dtype}>"

    text = ["module {", f"  func.func @{args.name}({sig}) {{"]
    text += b.lines
    text.append(f"    bufferization.materialize_in_destination {result} in restrict writable {dst}"
                f" : ({mlir_type(out_shape, out_dtype)},"
                f" memref<" + "x".join(map(str, out_shape)) + f"x{out_dtype}>) -> ()")
    text += ["    return", "  }", "}"]

    open(args.out, "w").write("\n".join(text) + "\n")
    print(f"wrote {args.out}")
    print(f"function parameters: {len(b.params)} (input + weights), plus one output buffer")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())