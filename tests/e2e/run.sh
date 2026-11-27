#!/usr/bin/env bash
# Copyright 2026 Indonesia Chip Design Collaborative Center (ICDeC)
# Author: Muhammad Faudzan Abdullah
# SPDX-License-Identifier: Apache-2.0
#
# One small model through the whole flow, from the trained model to numbers.
#
# tiny-cnn has the layer layout of the HAR model at a fraction of the size,
# with random weights. What the importer makes of it is kept in this directory
# (tiny.mlir and tiny_weights.*), so the compiler half of the flow runs without
# Python:
#
#   tiny.mlir -> C for the CPU and C for the accelerator -> run -> four logits
#
# The importer half, from the SavedModel to those three files, needs
# TensorFlow. It runs when python3 has it, and is reported as skipped when it
# does not.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$(dirname "$HERE")")"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
source "$ROOT/tests/strict-cflags.sh"

# Compile the model one way, run it, and check where the work went.
# Arguments: a short name, the compiler mode, the summary line expected, and
# how to call this path in the messages.
check_path() {
  local id="$1" mode="$2" summary="$3" what="$4"

  if ! "$ROOT/icdecompiler" "$HERE/tiny.mlir" $mode -o "$TMP/tiny_$id.c" \
       > /dev/null 2> "$TMP/tiny_$id.err"; then
    echo "FAIL: tiny-cnn does not compile $what" >&2
    grep -m3 'error:' "$TMP/tiny_$id.err" >&2 || true
    exit 1
  fi
  if grep -q 'warning:' "$TMP/tiny_$id.err"; then
    echo "FAIL: compiling tiny-cnn $what gives a warning" >&2
    grep 'warning:' "$TMP/tiny_$id.err" >&2
    exit 1
  fi
  if ! tr -s ' ' < "$TMP/tiny_$id.err" | grep -qF "$summary"; then
    echo "FAIL: tiny-cnn $what: the summary has no line '$summary'" >&2
    sed -n '/ICDeCompiler summary/,$p' "$TMP/tiny_$id.err" >&2
    exit 1
  fi

  # The accelerator path includes systolic_drv.h; tests/numeric has the host
  # stand-in for it, which does the same multiply in plain C.
  gcc $STRICT_CFLAGS -O2 -I"$HERE" -I"$ROOT/tests/numeric" \
      "$TMP/tiny_$id.c" "$HERE/tiny_weights.c" "$HERE/main_tiny.c" \
      -o "$TMP/tiny_$id"
  "$TMP/tiny_$id"
  echo "OK: tiny-cnn $what gives the expected logits"
}

check_path cpu --cpu "on CPU : 3552 (100%)"                     "on the CPU"
check_path acc ""    "on accelerator : 3552 (100%) in 5 calls"  "on the accelerator"

# The importer half. Its output has to be the files kept here, byte for byte,
# or the two halves no longer describe the same model.
if [[ -x "$ROOT/model-tools/.venv/bin/python3" ]]; then
  PATH="$ROOT/model-tools/.venv/bin:$PATH"
fi
if ! python3 -c "import numpy, tensorflow" > /dev/null 2>&1; then
  echo "SKIP: importer not checked, python3 has no TensorFlow (see Setup in model-tools/README.md)"
  exit 0
fi

# Run from model-tools with a relative path: the path of the model is written
# into the first line of the weight files.
if ! ( cd "$ROOT/model-tools" &&
       ./build_model.sh --q88 examples/tiny-cnn/savedmodel_tiny tiny -o "$TMP/regen" ) \
     > "$TMP/regen.log" 2>&1; then
  echo "FAIL: build_model.sh does not get through tiny-cnn" >&2
  tail -20 "$TMP/regen.log" >&2
  exit 1
fi
for f in tiny.mlir tiny_weights.c tiny_weights.h; do
  if ! cmp -s "$TMP/regen/$f" "$HERE/$f"; then
    echo "FAIL: the importer no longer produces tests/e2e/$f from the SavedModel" >&2
    diff "$HERE/$f" "$TMP/regen/$f" | head -6 >&2 || true
    exit 1
  fi
done
echo "OK: the importer produces tiny.mlir and the weights from the SavedModel"
