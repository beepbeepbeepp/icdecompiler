#!/usr/bin/env python3
"""
Extract INT8 weights and quantization parameters from a TFLite model
into C source suitable for bare-metal inference.

Emits <prefix>.c and <prefix>.h containing, per layer:
  - int8 weights and int32 biases, exactly as stored in the model
  - per-channel requantization multiplier (Q31) and right shift
  - input and output zero points as macros

Usage:
    python3 extract_weights.py model.tflite -o har_weights_int8
"""

import argparse
import json
import math

import numpy as np
import tflite


def quantize_multiplier(m):
    """Port of TFLite QuantizeMultiplier: m -> (int32 multiplier, right shift)."""
    if m == 0:
        return 0, 0
    frac, exp = math.frexp(m)                # m = frac * 2^exp, frac in [0.5, 1)
    fixed = int(round(frac * (1 << 31)))
    if fixed == (1 << 31):
        fixed //= 2
        exp += 1
    return fixed, -exp                       # stored as a right shift


class Model:
    def __init__(self, path):
        with open(path, "rb") as f:
            self.model = tflite.Model.GetRootAsModel(f.read(), 0)
        self.sg = self.model.Subgraphs(0)
        self.opnames = self._op_names()

    def _op_names(self):
        table = {getattr(tflite.BuiltinOperator, n): n
                 for n in dir(tflite.BuiltinOperator) if not n.startswith("_")}
        return [table.get(self.model.OperatorCodes(i).BuiltinCode(), "?")
                for i in range(self.model.OperatorCodesLength())]

    def tensor(self, idx):
        t = self.sg.Tensors(idx)
        q = t.Quantization()
        scales = np.array([q.Scale(i) for i in range(q.ScaleLength())], dtype=np.float64)
        zps = np.array([q.ZeroPoint(i) for i in range(q.ZeroPointLength())])
        buf = self.model.Buffers(t.Buffer())
        data = None
        if buf.DataLength():
            raw = buf.DataAsNumpy().tobytes()
            dtype = {0: np.float32, 2: np.int32, 9: np.int8}[t.Type()]
            shape = [t.Shape(j) for j in range(t.ShapeLength())]
            data = np.frombuffer(raw, dtype=dtype).reshape(shape)
        return data, scales, zps

    def weighted_layers(self):
        """Yield (kind, weight_idx, bias_idx, input_idx, output_idx) in model order."""
        for i in range(self.sg.OperatorsLength()):
            op = self.sg.Operators(i)
            kind = self.opnames[op.OpcodeIndex()]
            if kind not in ("CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED"):
                continue
            ins = [op.Inputs(j) for j in range(op.InputsLength())]
            bias = ins[2] if len(ins) > 2 else -1
            yield kind, ins[1], bias, ins[0], op.Outputs(0)


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
    ap.add_argument("model")
    ap.add_argument("-o", "--out", required=True, help="output prefix, no extension")
    ap.add_argument("--prefix", default=None, help="C symbol prefix (default: output name)")
    args = ap.parse_args()

    prefix = args.prefix or args.out.replace("-", "_")
    m = Model(args.model)

    src = [f"/* Generated from {args.model}. Do not edit by hand. */",
           "", f'#include "{args.out}.h"', ""]
    hdr = [f"/* Generated from {args.model}. Do not edit by hand. */",
           f"#ifndef {prefix.upper()}_H", f"#define {prefix.upper()}_H",
           "#include <stdint.h>", ""]
    summary = {}

    counters = {"CONV_2D": 0, "DEPTHWISE_CONV_2D": 0, "FULLY_CONNECTED": 0}
    for kind, wi, bi, ii, oi in m.weighted_layers():
        counters[kind] += 1
        base = {"CONV_2D": "conv", "DEPTHWISE_CONV_2D": "dwconv",
                "FULLY_CONNECTED": "fc"}[kind]
        name = f"{base}{counters[kind]}"

        w, w_scales, _ = m.tensor(wi)
        b = m.tensor(bi)[0] if bi >= 0 else None
        _, in_scales, in_zps = m.tensor(ii)
        _, out_scales, out_zps = m.tensor(oi)

        # requantization multiplier per output channel
        mult = (in_scales[0] * w_scales) / out_scales[0]
        m0, shift = zip(*(quantize_multiplier(v) for v in mult))

            # dense weights are stored [out, in]; transpose to [1, in, out]
        if kind == "FULLY_CONNECTED":
            w_out = w.T[None]
        elif kind == "DEPTHWISE_CONV_2D":
            # TFLite [1, KH, KW, C*M] -> TOSA [KH, KW, C, M]; the bytes already
            # sit in that order, only the declared shape changes
            c_in = int(m.sg.Tensors(ii).ShapeAsNumpy()[-1])
            w_out = w.reshape(w.shape[1], w.shape[2], c_in, w.shape[3] // c_in)
        else:
            w_out = w

        src += [c_array("int8_t", f"{prefix}_{name}_w", w_out, dims_of(w_out.shape), 16), ""]
        if b is not None:
            src += [c_array("int32_t", f"{prefix}_{name}_b", b, dims_of(b.shape), 8), ""]
        src += [c_array("int32_t", f"{prefix}_{name}_m0", np.array(m0), f"[{len(m0)}]", 8), ""]
        src += [c_array("int32_t", f"{prefix}_{name}_shift", np.array(shift), f"[{len(shift)}]", 8), ""]

        hdr += [f"extern const int8_t  {prefix}_{name}_w{dims_of(w_out.shape)};"]
        if b is not None:
            hdr += [f"extern const int32_t {prefix}_{name}_b{dims_of(b.shape)};"]
        hdr += [f"extern const int32_t {prefix}_{name}_m0[{len(m0)}];",
                f"extern const int32_t {prefix}_{name}_shift[{len(shift)}];",
                f"#define {prefix.upper()}_{name.upper()}_IN_ZP  ({int(in_zps[0])})",
                f"#define {prefix.upper()}_{name.upper()}_OUT_ZP ({int(out_zps[0])})",
                ""]

        summary[name] = dict(kind=kind, w_shape=list(w_out.shape),
                             in_zp=int(in_zps[0]), out_zp=int(out_zps[0]),
                             in_scale=float(in_scales[0]), out_scale=float(out_scales[0]),
                             mult=mult.tolist())
        print(f"{name:8} {kind:16} w{list(w_out.shape)}  "
              f"in_zp={in_zps[0]:5} out_zp={out_zps[0]:5}  "
              f"mult=[{mult.min():.4g}, {mult.max():.4g}]")

    hdr += ["#endif"]
    open(args.out + ".c", "w").write("\n".join(src))
    open(args.out + ".h", "w").write("\n".join(hdr))
    json.dump(summary, open(args.out + ".json", "w"), indent=1)
    print(f"\nwrote {args.out}.c, {args.out}.h, {args.out}.json")


if __name__ == "__main__":
    main()