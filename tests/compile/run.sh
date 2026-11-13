#!/usr/bin/env bash
# Copyright 2026 Indonesia Chip Design Collaborative Center (ICDeC)
# Author: Muhammad Faudzan Abdullah
# SPDX-License-Identifier: Apache-2.0
#
# Verify generated C compiles against the real systolic driver header.
# Uses a stub pulp.h so this can run on the host without the PULP SDK.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$(dirname "$HERE")")"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

INPUT="${1:-$ROOT/tests/inputs/int_matmul.mlir}"

"$ROOT/icdecompiler" "$INPUT" -o "$TMP/body.c" > /dev/null 2>&1

{
  echo '#include <stdint.h>'
  echo '#include <stddef.h>'
  echo '#include "systolic_drv.h"'
  cat "$TMP/body.c"
} > "$TMP/full.c"

gcc -c "$TMP/full.c" -I"$ROOT" -I"$HERE/stub-include" -o "$TMP/out.o"

echo "OK: generated C compiles and links against systolic_drv.h"
nm "$TMP/out.o" | grep -q systolic_multiply_large \
  || { echo "FAIL: systolic_multiply_large missing from object" >&2; exit 1; }
echo "OK: systolic_multiply_large present in object"

# Ops the accelerator cannot take stay on the CPU with a warning, and the rest
# of the model still compiles. Arguments: input name, accelerator calls
# expected, text the warning must contain.
check_stays_on_cpu() {
  local name="$1" want_calls="$2" warning="$3" calls
  "$ROOT/icdecompiler" "$ROOT/tests/inputs/$name.mlir" -o "$TMP/$name.c" \
    > /dev/null 2> "$TMP/$name.err"
  calls=$(grep -c systolic_multiply_large "$TMP/$name.c" || true)
  if [[ "$calls" != "$want_calls" ]]; then
    echo "FAIL: $name: $calls accelerator calls, expected $want_calls" >&2
    exit 1
  fi
  if ! grep -q "$warning" "$TMP/$name.err"; then
    echo "FAIL: $name: no warning containing '$warning'" >&2
    exit 1
  fi
  if grep -q 'malloc\|free' "$TMP/$name.c"; then
    echo "FAIL: $name: generated C uses malloc or free" >&2
    exit 1
  fi
  echo "OK: $name stays on the CPU and warns"
}

# K above what the bitstream takes
check_stays_on_cpu matmul_k_limit 1 "exceeds the accelerator limit"
check_stays_on_cpu conv_k_limit   0 "exceeds the accelerator limit"
# element types other than i16 x i16 -> i32
check_stays_on_cpu matmul_i8      0 "matmul is not i16 x i16 -> i32"
check_stays_on_cpu conv_i8        0 "convolution is not i16 x i16 -> i32"
check_stays_on_cpu simple_matmul  0 "matmul is not i16 x i16 -> i32"
# a batch larger than one
check_stays_on_cpu matmul_batch2  0 "matmul has a batch larger than one"
check_stays_on_cpu conv_batch2    0 "convolution has a batch larger than one"
