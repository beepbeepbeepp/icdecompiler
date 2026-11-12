#!/usr/bin/env python3
# Copyright 2026 Indonesia Chip Design Collaborative Center (ICDeC)
# Author: Muhammad Faudzan Abdullah
# SPDX-License-Identifier: Apache-2.0

"""
Generate TOSA MLIR in Q8.8 fixed point from a float model's weights.

Q8.8 is what the systolic array driver expects: int16 values with eight
fractional bits, accumulated in int32. Starting from float weights keeps the
conversion to a single rounding step.

Layer shapes are read from the weights themselves; the architecture file only
supplies what the weights cannot tell us, such as stride, padding, and where
pooling happens. BatchNorm is folded into the preceding convolution by
extract_weights_q88.py, so it never appears here.

Adding a layer type means writing one function and registering it in HANDLERS.

Usage:
    python3 keras_to_tosa_q88.py savedmodelhar -o har_q88.mlir --name har_model
"""

import argparse

import numpy as np
from savedmodel_layers import layer_names, batchnorm_for


# A Q16.16 accumulator shifted back to Q8.8 is an exact >> 8. TOSA spells that
# as a rescale with multiplier 2^30 and shift 38.
RESCALE_MULT = 1073741824
RESCALE_SHIFT = 38

# What the weights cannot tell us: order, stride, padding, pooling.
ARCH = [
    ("conv1d",    dict(stride=2, pad="same", relu=True)),
    ("maxpool1d", dict(pool=2)),
    ("conv1d",    dict(stride=2, pad="same", relu=True)),
    ("maxpool1d", dict(pool=2)),
    ("conv1d",    dict(stride=1, pad="same", relu=True)),
    ("gap",       dict()),
    ("dense",     dict(relu=True)),
    ("dense",     dict(relu=False)),
]

INPUT_LENGTH = 512


def mlir_type(shape, dtype):
    return "tensor<" + "x".join(str(d) for d in shape) + f"x{dtype}>"


def same_padding(in_size, k, stride):
    out = (in_size + stride - 1) // stride
    total = max(0, (out - 1) * stride + k - in_size)
    return total // 2, total - total // 2


class Builder:
    def __init__(self):
        self.lines = []
        self.params = []
        self._n = 0

    def fresh(self, hint="v"):
        self._n += 1
        return f"%{hint}{self._n}"

    def emit(self, line):
        self.lines.append("    " + line)

    def param(self, shape, dtype, hint="p"):
        n = self.fresh(hint)
        self.params.append((n, mlir_type(shape, dtype)))
        return n

    def const(self, values, shape, dtype):
        n = self.fresh("c")
        body = f"dense<{values[0]}>" if len(values) == 1 \
            else "dense<[" + ", ".join(str(v) for v in values) + "]>"
        ty = mlir_type(shape, dtype)
        self.emit(f'{n} = "tosa.const"() {{values = {body} : {ty}}} : () -> {ty}')
        return n

    def const_shape(self, shape):
        n = self.fresh("s")
        self.emit(f"{n} = tosa.const_shape {{values = dense<[{', '.join(map(str, shape))}]>"
                  f" : tensor<{len(shape)}xindex>}} : () -> !tosa.shape<{len(shape)}>")
        return n

    def to_q88(self, src, shape):
        """Shift a Q16.16 accumulator back down to Q8.8."""
        n_mul = self.const([RESCALE_MULT], [1], "i32")
        n_shf = self.const([RESCALE_SHIFT], [1], "i8")
        n_izp = self.const([0], [1], "i32")
        n_ozp = self.const([0], [1], "i16")
        n = self.fresh("r")
        self.emit(
            f"{n} = tosa.rescale {src}, {n_mul}, {n_shf}, {n_izp}, {n_ozp} {{"
            f" scale32 = true,"
            f" rounding_mode = #tosa.rounding_mode<SINGLE_ROUND>,"
            f" per_channel = false,"
            f" input_unsigned = false, output_unsigned = false }}"
            f" : ({mlir_type(shape, 'i32')}, tensor<1xi32>, tensor<1xi8>,"
            f" tensor<1xi32>, tensor<1xi16>) -> {mlir_type(shape, 'i16')}")
        return n

    def relu(self, src, shape):
        n = self.fresh("v")
        ty = mlir_type(shape, "i16")
        self.emit(f"{n} = tosa.clamp {src} {{ min_val = 0 : i16,"
                  f" max_val = 32767 : i16 }} : ({ty}) -> {ty}")
        return n


HANDLERS = {}


def handler(kind):
    def wrap(fn):
        HANDLERS[kind] = fn
        return fn
    return wrap

@handler("conv1d")
def _conv1d(b, st, cfg):
    """Conv1D as TOSA conv2d with height 1, then rescale back to Q8.8."""
    w, _ = st["convs"].pop(0)
    K, Cin, F = w.shape
    L = st["length"]

    if cfg["pad"] == "same":
        pl, pr = same_padding(L, K, cfg["stride"])
    else:
        pl = pr = 0
    Lo = (L + pl + pr - K) // cfg["stride"] + 1

    in_shape = [1, 1, L, Cin]
    w_shape = [F, 1, K, Cin]
    out_shape = [1, 1, Lo, F]

    n_w = b.param(w_shape, "i16", "w")
    n_b = b.param([F], "i32", "b")
    n_izp = b.const([0], [1], "i16")
    n_wzp = b.const([0], [1], "i16")

    n_acc = b.fresh("a")
    b.emit(f"{n_acc} = tosa.conv2d {st['cur']}, {n_w}, {n_b}, {n_izp}, {n_wzp} {{"
           f" acc_type = i32,"
           f" dilation = array<i64: 1, 1>,"
           f" pad = array<i64: 0, 0, {pl}, {pr}>,"
           f" stride = array<i64: 1, {cfg['stride']}> }}"
           f" : ({mlir_type(in_shape, 'i16')}, {mlir_type(w_shape, 'i16')},"
           f" {mlir_type([F], 'i32')}, tensor<1xi16>, tensor<1xi16>)"
           f" -> {mlir_type(out_shape, 'i32')}")

    n = b.to_q88(n_acc, out_shape)
    if cfg.get("relu"):
        n = b.relu(n, out_shape)

    st["cur"] = n
    st["length"] = Lo
    st["channels"] = F


@handler("maxpool1d")
def _maxpool1d(b, st, cfg):
    p = cfg["pool"]
    L, C = st["length"], st["channels"]
    in_shape = [1, 1, L, C]
    out_shape = [1, 1, L // p, C]

    n = b.fresh("v")
    b.emit(f"{n} = tosa.max_pool2d {st['cur']} {{"
           f" kernel = array<i64: 1, {p}>,"
           f" pad = array<i64: 0, 0, 0, 0>,"
           f" stride = array<i64: 1, {p}> }}"
           f" : ({mlir_type(in_shape, 'i16')}) -> {mlir_type(out_shape, 'i16')}")

    st["cur"] = n
    st["length"] = L // p


@handler("gap")
def _gap(b, st, cfg):
    """Average over length. Reshape to 3-D first so the reduction lands
    directly on the shape the dense layers expect."""
    L, C = st["length"], st["channels"]

    # [1,1,L,C] -> [1,L,C] before reducing, so no reshape is needed after
    n_shape3 = b.const_shape([1, L, C])
    n_3d = b.fresh("v")
    b.emit(f"{n_3d} = tosa.reshape {st['cur']}, {n_shape3}"
           f" : ({mlir_type([1, 1, L, C], 'i16')}, !tosa.shape<3>)"
           f" -> {mlir_type([1, L, C], 'i16')}")

    n_wide = b.fresh("v")
    b.emit(f"{n_wide} = tosa.cast {n_3d}"
           f" : ({mlir_type([1, L, C], 'i16')}) -> {mlir_type([1, L, C], 'i32')}")

    n_sum = b.fresh("a")
    b.emit(f"{n_sum} = tosa.reduce_sum {n_wide} {{axis = 1 : i32}}"
           f" : ({mlir_type([1, L, C], 'i32')}) -> {mlir_type([1, 1, C], 'i32')}")

    # dividing by L is a rescale whose multiplier is 2^31 / L
    mult = int(round((1 << 31) / L))
    n_mul = b.const([mult], [1], "i32")
    n_shf = b.const([31], [1], "i8")
    n_izp = b.const([0], [1], "i32")
    n_ozp = b.const([0], [1], "i16")
    n = b.fresh("r")
    b.emit(f"{n} = tosa.rescale {n_sum}, {n_mul}, {n_shf}, {n_izp}, {n_ozp} {{"
           f" scale32 = true,"
           f" rounding_mode = #tosa.rounding_mode<SINGLE_ROUND>,"
           f" per_channel = false,"
           f" input_unsigned = false, output_unsigned = false }}"
           f" : ({mlir_type([1, 1, C], 'i32')}, tensor<1xi32>, tensor<1xi8>,"
           f" tensor<1xi32>, tensor<1xi16>) -> {mlir_type([1, 1, C], 'i16')}")

    st["cur"] = n
    st["length"] = 1


@handler("dense")
def _dense(b, st, cfg):
    w, _ = st["denses"].pop(0)
    Cin, Cout = w.shape

    n_w = b.param([1, Cin, Cout], "i16", "w")
    n_b = b.param([Cout], "i32", "b")
    n_azp = b.const([0], [1], "i16")
    n_bzp = b.const([0], [1], "i16")

    acc_shape = [1, 1, Cout]
    n_mm = b.fresh("a")
    b.emit(f"{n_mm} = tosa.matmul {st['cur']}, {n_w}, {n_azp}, {n_bzp}"
           f" : ({mlir_type([1, 1, Cin], 'i16')}, {mlir_type([1, Cin, Cout], 'i16')},"
           f" tensor<1xi16>, tensor<1xi16>) -> {mlir_type(acc_shape, 'i32')}")

    # bias is Q16.16, matching the accumulator
    n_bshape = b.const_shape([1, 1, Cout])
    n_br = b.fresh("v")
    b.emit(f"{n_br} = tosa.reshape {n_b}, {n_bshape}"
           f" : ({mlir_type([Cout], 'i32')}, !tosa.shape<3>)"
           f" -> {mlir_type(acc_shape, 'i32')}")
    n_add = b.fresh("a")
    b.emit(f"{n_add} = tosa.add {n_mm}, {n_br}"
           f" : ({mlir_type(acc_shape, 'i32')}, {mlir_type(acc_shape, 'i32')})"
           f" -> {mlir_type(acc_shape, 'i32')}")

    n = b.to_q88(n_add, acc_shape)
    if cfg.get("relu"):
        n = b.relu(n, acc_shape)

    st["cur"] = n
    st["channels"] = Cout

def input_length(path):
    """Sequence length the SavedModel expects, read from its serving signature."""
    import tensorflow as tf
    for sig in tf.saved_model.load(path).signatures.values():
        for spec in sig.structured_input_signature[1].values():
            if spec.shape.rank == 3 and spec.shape[1] is not None:
                return int(spec.shape[1])
    return None

def load_weights(path):
    """Read variables from a SavedModel, folding BatchNorm into each conv."""
    import tensorflow as tf
    V = {v.name: v.numpy() for v in tf.saved_model.load(path).variables}

    convs, denses = [], []
    for name in layer_names(V, "conv1d"):
        w = V[f"{name}/kernel:0"]
        b = V[f"{name}/bias:0"]
        bn = batchnorm_for(name, "conv1d")
        if f"{bn}/gamma:0" in V:
            s = V[f"{bn}/gamma:0"] / np.sqrt(V[f"{bn}/moving_variance:0"] + 1e-3)
            w = w * s
            b = (b - V[f"{bn}/moving_mean:0"]) * s + V[f"{bn}/beta:0"]
        convs.append((w, b))
    for name in layer_names(V, "dense"):
        denses.append((V[f"{name}/kernel:0"], V[f"{name}/bias:0"]))
    return convs, denses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="SavedModel directory")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--name", default="model")
    ap.add_argument("--length", type=int, default=None, help="input length; read from the model when omitted")
    args = ap.parse_args()

    if args.length is None:
        args.length = input_length(args.model) or INPUT_LENGTH
    print(f"input length {args.length}")

    convs, denses = load_weights(args.model)
    print(f"{len(convs)} conv layers, {len(denses)} dense layers")

    missing = sorted({k for k, _ in ARCH if k not in HANDLERS})
    if missing:
        print("unsupported layers:", ", ".join(missing))
        return 1

    b = Builder()
    channels = convs[0][0].shape[1]
    n_in = b.param([1, 1, args.length, channels], "i16", "input")

    st = {"cur": n_in, "length": args.length, "channels": channels,
          "convs": list(convs), "denses": list(denses)}

    for kind, cfg in ARCH:
        HANDLERS[kind](b, st, cfg)

    out_shape = [1, 1, st["channels"]]
    dst = b.fresh("dst")
    sig = ", ".join(f"{n}: {t}" for n, t in b.params)
    sig += f", {dst}: memref<" + "x".join(map(str, out_shape)) + "xi16>"

    text = ["module {", f"  func.func @{args.name}({sig}) {{"]
    text += b.lines
    text.append(f"    bufferization.materialize_in_destination {st['cur']}"
                f" in restrict writable {dst}"
                f" : ({mlir_type(out_shape, 'i16')},"
                f" memref<" + "x".join(map(str, out_shape)) + "xi16>) -> ()")
    text += ["    return", "  }", "}"]

    open(args.out, "w").write("\n".join(text) + "\n")
    print(f"wrote {args.out}")
    print(f"function parameters: {len(b.params)} (input + weights), plus one output buffer")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())