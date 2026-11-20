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

# Stop before the assembler: the driver has RISC-V instructions (fence) that a
# host assembler rejects. Type and prototype checks all happen before that.
gcc -S "$TMP/full.c" -I"$ROOT" -I"$HERE/stub-include" -o "$TMP/out.s"

echo "OK: generated C compiles against systolic_drv.h"
grep -q 'call.*systolic_multiply_large_hwpe' "$TMP/out.s" \
  || { echo "FAIL: no call to systolic_multiply_large_hwpe in the output" >&2; exit 1; }
echo "OK: generated C calls systolic_multiply_large_hwpe"

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

# A linalg.generic that is not a plain signed matmul must not reach the
# accelerator. Argument: input name.
check_not_offloaded() {
  local name="$1" calls
  "$ROOT/icdecompiler" "$ROOT/tests/inputs/$name.mlir" -o "$TMP/$name.c" \
    > /dev/null 2> "$TMP/$name.err"
  calls=$(grep -c systolic_multiply_large "$TMP/$name.c" || true)
  if [[ "$calls" != 0 ]]; then
    echo "FAIL: $name: $calls accelerator calls, expected 0" >&2
    exit 1
  fi
  echo "OK: $name is not offloaded"
}

check_not_offloaded generic_a_transposed
check_not_offloaded generic_body_square
check_not_offloaded generic_body_unsigned
check_not_offloaded generic_body_add_twice

# A rewrite that cannot finish must leave nothing behind. Here the second
# operand cannot become a C pointer, which the rewrite used to find out only
# after it had created ops for the first one.
OPT="$ROOT/build/bin/icdec-opt"
[[ -x "$OPT" ]] || OPT="$ROOT/icdec-opt"
left=$("$OPT" "$ROOT/tests/inputs/matmul_late_fail.mlir" \
         --icdec-matmul-to-systolic 2> /dev/null | grep -c 'emitc\.' || true)
if [[ "$left" != 0 ]]; then
  echo "FAIL: matmul_late_fail: a failed rewrite left $left emitc ops behind" >&2
  exit 1
fi
echo "OK: matmul_late_fail leaves the IR untouched"

# An operand that is a slice of a larger buffer is not M x K elements in a
# row, which is what the driver reads. It must stay on the CPU, and the model
# must still compile.
check_not_offloaded matmul_slice_offset
check_not_offloaded matmul_slice_stride
check_not_offloaded matmul_subview_cols
check_not_offloaded matmul_subview_rows
