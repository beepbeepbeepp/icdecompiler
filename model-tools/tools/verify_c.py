#!/usr/bin/env python3
# Copyright 2026 Indonesia Chip Design Collaborative Center (ICDeC)
# Author: Muhammad Faudzan Abdullah
# SPDX-License-Identifier: Apache-2.0

"""
Compare a C inference implementation against the TFLite interpreter.

The C side must be built into a program that reads the quantized input as
whitespace-separated integers on stdin and prints the output logits the same
way. See har_test in the model-tools README for the expected shape.

Usage:
    python3 verify_c.py model.tflite ./har_test --input-shape 512 3 \
        --output-tensor 35 --trials 30
"""

import argparse
import subprocess

import numpy as np
from ai_edge_litert.interpreter import Interpreter


def synth_accel(rng, length):
    """A signal shaped roughly like a wrist accelerometer window."""
    t = np.linspace(0, 4 * np.pi * rng.uniform(0.5, 3), length)
    sig = np.stack([
        9.8 * rng.uniform(-0.2, 0.2) + rng.uniform(0.5, 3) * np.sin(t + rng.uniform(0, 6)),
        rng.uniform(0.5, 3) * np.sin(1.7 * t + rng.uniform(0, 6)),
        9.8 + rng.uniform(0.5, 3) * np.sin(0.6 * t + rng.uniform(0, 6)),
    ], axis=-1).astype(np.float32)
    return sig + rng.normal(0, 0.1, sig.shape).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("program", help="C test binary reading stdin, printing logits")
    ap.add_argument("--input-shape", type=int, nargs="+", required=True)
    ap.add_argument("--output-tensor", type=int, default=None,
                    help="tensor index to compare against; default is the model output")
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    interp = Interpreter(model_path=args.model)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out_idx = args.output_tensor
    if out_idx is None:
        out_idx = interp.get_output_details()[0]["index"]
    scale, zp = inp["quantization"]

    rng = np.random.default_rng(args.seed)
    exact = same_class = 0
    max_abs = 0

    for trial in range(args.trials):
        sig = synth_accel(rng, args.input_shape[0])
        xq = np.clip(np.round(sig / scale + zp), -128, 127).astype(np.int8)

        interp.set_tensor(inp["index"], xq[None])
        interp.invoke()
        ref = interp.get_tensor(out_idx).ravel()

        proc = subprocess.run([args.program], input=" ".join(map(str, xq.ravel())),
                              capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"trial {trial}: program failed\n{proc.stderr}")
            return 1
        mine = np.array(list(map(int, proc.stdout.split())), dtype=np.int32)

        if mine.shape != ref.shape:
            print(f"trial {trial}: shape mismatch, got {mine.shape}, expected {ref.shape}")
            return 1

        diff = np.abs(mine.astype(int) - ref.astype(int))
        max_abs = max(max_abs, int(diff.max()))
        exact += int(diff.max() == 0)
        same_class += int(mine.argmax() == ref.argmax())

        if trial < 3:
            print(f"[{trial}] C={mine} tflite={ref}")

    n = args.trials
    print(f"\nbit-exact       : {exact}/{n}")
    print(f"class agreement : {same_class}/{n}")
    print(f"max abs diff    : {max_abs}")
    return 0 if same_class == n else 1


if __name__ == "__main__":
    raise SystemExit(main())