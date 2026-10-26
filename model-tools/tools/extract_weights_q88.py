#!/usr/bin/env python3
"""
Extract Q8.8 weights from a float Keras SavedModel into C source.

Q8.8 is what the systolic array driver expects: int16 inputs, int32
accumulators. Starting from the float model avoids the double rounding that
converting from an INT8 model would introduce.

BatchNorm layers are folded into the preceding convolution, since the array
has no notion of normalization.

Weights and activations are Q8.8 (int16). Biases are Q16.16 (int32) so they can
be added straight into the accumulator without a shift. Final logits stay int32
because they exceed the Q8.8 range and are only used for argmax.

Usage:
    python3 extract_weights_q88.py path/to/savedmodel -o har_weights_q88 --prefix har
"""

import argparse
import os

import numpy as np
from savedmodel_layers import layer_names, batchnorm_for


Q88_SCALE = 256          # 2^8
Q1616_SCALE = 65536      # 2^16


def to_q88(x):
    """Round a float array to Q8.8, saturating at the int16 range."""
    q = np.round(np.asarray(x) * Q88_SCALE)
    over = int(np.sum((q > 32767) | (q < -32768)))
    return np.clip(q, -32768, 32767).astype(np.int16), over


def to_q1616(x):
    q = np.round(np.asarray(x) * Q1616_SCALE)
    over = int(np.sum((q > 2**31 - 1) | (q < -2**31)))
    return np.clip(q, -(2**31), 2**31 - 1).astype(np.int32), over


def load_variables(path):
    """Read every variable from a SavedModel, keyed by name."""
    import tensorflow as tf
    model = tf.saved_model.load(path)
    return {v.name: v.numpy() for v in model.variables}


def fold_batchnorm(w, b, gamma, beta, mean, var, eps=1e-3):
    """Fold BatchNorm into the preceding conv or dense layer."""
    scale = gamma / np.sqrt(var + eps)
    return w * scale, (b - mean) * scale + beta


def collect_layers(V):
    """Pair each conv with its BatchNorm, then append the dense layers.

    A conv and its BatchNorm share the same name suffix, which is how Keras
    builds a functional model: conv1d_192 goes with batch_normalization_192.
    """
    layers = []
    for i, name in enumerate(layer_names(V, "conv1d"), start=1):
        w = V[f"{name}/kernel:0"]          # [K, Cin, F]
        b = V[f"{name}/bias:0"]
        bn = batchnorm_for(name, "conv1d")
        if f"{bn}/gamma:0" in V:
            w, b = fold_batchnorm(w, b,
                                  V[f"{bn}/gamma:0"], V[f"{bn}/beta:0"],
                                  V[f"{bn}/moving_mean:0"],
                                  V[f"{bn}/moving_variance:0"])
        layers.append((f"conv{i}", "conv", w, b))

    for i, name in enumerate(layer_names(V, "dense"), start=1):
        layers.append((f"fc{i}",
                       "dense",
                       V[f"{name}/kernel:0"],     # [Cin, Cout]
                       V[f"{name}/bias:0"]))
    return layers


def c_array(ctype, name, arr, dims, per_line):
    flat = arr.ravel()
    lines = [f"const {ctype} {name}{dims} = {{"]
    for i in range(0, len(flat), per_line):
        chunk = ", ".join(str(int(v)) for v in flat[i:i + per_line])
        lines.append("    " + chunk + ("," if i + per_line < len(flat) else ""))
    lines.append("};")
    return "\n".join(lines)


def dims_of(shape):
    return "".join(f"[{d}]" for d in shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="path to a Keras SavedModel directory")
    ap.add_argument("-o", "--out", required=True, help="output prefix, no extension")
    ap.add_argument("--prefix", default=None, help="C symbol prefix")
    args = ap.parse_args()

    prefix = args.prefix or args.out.replace("-", "_")
    V = load_variables(args.model)
    layers = collect_layers(V)

    src = [f"/* Q8.8 weights generated from {args.model}. Do not edit by hand. */",
            "", f'#include "{os.path.basename(args.out)}.h"', ""]
    hdr = [f"/* Q8.8 weights generated from {args.model}. Do not edit by hand. */",
           f"#ifndef {prefix.upper()}_Q88_H", f"#define {prefix.upper()}_Q88_H",
           "#include <stdint.h>", "",
           "/* Weights and activations are Q8.8; biases are Q16.16 so they add",
           "   directly into the accumulator. */", ""]

    total_overflow = 0
    for name, kind, w, b in layers:
        # conv weights arrive as [K, Cin, F]; the array wants [F, K, Cin]
        # conv weights arrive as [K, Cin, F]; TOSA conv2d wants [F, 1, K, Cin]
        if kind == "conv":
            w_out = np.transpose(w, (2, 0, 1))[:, None, :, :]
        else:
            w_out = w[None]          # dense: [Cin, Cout] -> [1, Cin, Cout]

        wq, ow = to_q88(w_out)
        bq, ob = to_q1616(b)
        total_overflow += ow + ob

        src += [c_array("int16_t", f"{prefix}_{name}_w", wq, dims_of(wq.shape), 12), ""]
        src += [c_array("int32_t", f"{prefix}_{name}_b", bq, dims_of(bq.shape), 8), ""]

        hdr += [f"extern const int16_t {prefix}_{name}_w{dims_of(wq.shape)};",
                f"extern const int32_t {prefix}_{name}_b{dims_of(bq.shape)};"]

        flag = f"  ({ow + ob} saturated)" if ow + ob else ""
        print(f"{name:6} {kind:6} w{list(wq.shape)}  "
              f"float [{w_out.min():7.4f}, {w_out.max():7.4f}]  "
              f"q8.8 [{wq.min():6}, {wq.max():6}]{flag}")

    hdr += ["", "#endif"]
    open(args.out + ".c", "w").write("\n".join(src))
    open(args.out + ".h", "w").write("\n".join(hdr))

    print(f"\nwrote {args.out}.c, {args.out}.h")
    if total_overflow:
        print(f"WARNING: {total_overflow} values saturated at the Q8.8 range")
    else:
        print("no saturation: every weight fits Q8.8")


if __name__ == "__main__":
    main()