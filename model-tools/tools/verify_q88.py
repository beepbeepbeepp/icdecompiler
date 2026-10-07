#!/usr/bin/env python3
"""
Compare a Q8.8 C implementation against the float model it came from.

The C program must read the Q8.8 input as whitespace-separated integers on
stdin and print the output logits the same way. test_host_q88.c does that.

Usage:
    python3 verify_q88.py savedmodelhar ./har_q88_test --trials 30
"""

import argparse
import subprocess

import numpy as np
from savedmodel_layers import layer_names, batchnorm_for


def load_folded(path):
    """Read SavedModel variables and fold BatchNorm into each conv."""
    import tensorflow as tf
    V = {v.name: v.numpy() for v in tf.saved_model.load(path).variables}

    convs, denses = [], []
    for name in layer_names(V, "conv1d"):
        w, b = V[f"{name}/kernel:0"], V[f"{name}/bias:0"]
        bn = batchnorm_for(name, "conv1d")
        if f"{bn}/gamma:0" in V:
            s = V[f"{bn}/gamma:0"] / np.sqrt(V[f"{bn}/moving_variance:0"] + 1e-3)
            w, b = w * s, (b - V[f"{bn}/moving_mean:0"]) * s + V[f"{bn}/beta:0"]
        convs.append((w, b))
    for name in layer_names(V, "dense"):
        denses.append((V[f"{name}/kernel:0"], V[f"{name}/bias:0"]))
    return convs, denses


def conv1d(x, w, b, stride):
    K, C, F = w.shape
    Lo = (x.shape[0] + stride - 1) // stride
    need = max(0, (Lo - 1) * stride + K - x.shape[0])
    xp = np.pad(x, ((need // 2, need - need // 2), (0, 0)))
    y = np.empty((Lo, F), np.float32)
    for i in range(Lo):
        y[i] = np.einsum('kc,kcf->f', xp[i * stride:i * stride + K], w) + b
    return y


def run_float(x, convs, denses):
    relu = lambda v: np.maximum(v, 0)

    def mp(v, s):
        Lo = v.shape[0] // s
        return v[:Lo * s].reshape(Lo, s, -1).max(1)

    h = relu(conv1d(x, *convs[0], 2)); h = mp(h, 2)
    h = relu(conv1d(h, *convs[1], 2)); h = mp(h, 2)
    h = relu(conv1d(h, *convs[2], 1))
    g = h.mean(0)
    f = relu(g @ denses[0][0] + denses[0][1])
    return f @ denses[1][0] + denses[1][1]


def synth_accel(rng, length):
    t = np.linspace(0, 4 * np.pi * rng.uniform(0.5, 3), length)
    sig = np.stack([
        rng.uniform(0.5, 3) * np.sin(t + rng.uniform(0, 6)),
        rng.uniform(0.5, 3) * np.sin(1.7 * t + rng.uniform(0, 6)),
        rng.uniform(0.5, 3) * np.sin(0.6 * t + rng.uniform(0, 6)),
    ], axis=-1).astype(np.float32)
    sig += rng.normal(0, 0.1, sig.shape).astype(np.float32)
    return sig - sig.mean(axis=0, keepdims=True)     # same preprocessing as training


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="SavedModel directory")
    ap.add_argument("program", help="C test binary")
    ap.add_argument("--length", type=int, default=512)
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()

    convs, denses = load_folded(args.model)
    rng = np.random.default_rng(args.seed)

    agree = 0
    for trial in range(args.trials):
        sig = synth_accel(rng, args.length)

        ref = run_float(sig, convs, denses)

        xq = np.clip(np.round(sig * 256), -32768, 32767).astype(np.int16)
        proc = subprocess.run([args.program], input=" ".join(map(str, xq.ravel())),
                              capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"trial {trial}: program failed\n{proc.stderr}")
            return 1
        mine = np.array(list(map(int, proc.stdout.split())), dtype=np.int32)

        same = mine.argmax() == ref.argmax()
        agree += same
        if trial < 5:
            print(f"[{trial}] q8.8 argmax={mine.argmax()} float argmax={ref.argmax()}"
                  f"  {'ok' if same else 'DIFF'}   q8.8={mine}")

    print(f"\nclass agreement: {agree}/{args.trials}")
    return 0 if agree == args.trials else 1


if __name__ == "__main__":
    raise SystemExit(main())