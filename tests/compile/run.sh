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

# Ops the accelerator cannot take stay on the CPU with a warning that says why,
# and the rest of the model still compiles. No multiply-accumulate may go to
# the CPU in silence. Arguments: input name, accelerator calls expected, text
# the warning must contain.
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
# accelerator.
check_stays_on_cpu generic_a_transposed   0 "operand stored transposed"
check_stays_on_cpu generic_body_square    0 "not as acc + a \* b on signed values"
check_stays_on_cpu generic_body_unsigned  0 "not as acc + a \* b on signed values"
check_stays_on_cpu generic_body_add_twice 0 "not as acc + a \* b on signed values"

# The named ops have the same two ways out of a plain signed matmul: indexing
# maps that transpose an operand, and a cast that widens it as unsigned.
check_stays_on_cpu matmul_named_transposed       0 "operand stored transposed or broadcast"
check_stays_on_cpu matmul_named_unsigned         0 "matmul multiplies unsigned values"
check_stays_on_cpu batch_matmul_named_transposed 0 "operand stored transposed or broadcast"

# other ops that multiply and accumulate but are not a plain matmul
check_stays_on_cpu matmul_zero_point 0 "matmul has a zero point"
check_stays_on_cpu matvec            0 "linalg.matvec is not a matrix multiply"

# convolutions that im2col does not turn into a matmul the driver can take
check_stays_on_cpu conv_zero_point 0 "convolution has a zero point"
check_stays_on_cpu conv_dilation   0 "convolution has a dilation other than one"
check_stays_on_cpu conv_depthwise  0 "depthwise convolution is one small matmul"

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
check_stays_on_cpu matmul_slice_offset 0 "operand that is a slice of a larger buffer"
check_stays_on_cpu matmul_slice_stride 0 "operand that is a slice of a larger buffer"
check_stays_on_cpu matmul_subview_cols 0 "operand that is a slice of a larger buffer"
check_stays_on_cpu matmul_subview_rows 0 "operand that is a slice of a larger buffer"

# The other way round: work that does go to the accelerator gets no warning.
# Arguments: input name, accelerator calls expected.
check_offloaded_quietly() {
  local name="$1" want_calls="$2" calls
  "$ROOT/icdecompiler" "$ROOT/tests/inputs/$name.mlir" -o "$TMP/$name.c" \
    > /dev/null 2> "$TMP/$name.err"
  calls=$(grep -c 'systolic_multiply_large_hwpe(' "$TMP/$name.c" || true)
  if [[ "$calls" != "$want_calls" ]]; then
    echo "FAIL: $name: $calls accelerator calls, expected $want_calls" >&2
    exit 1
  fi
  if grep -q 'warning:' "$TMP/$name.err"; then
    echo "FAIL: $name: warns although everything is offloaded" >&2
    grep 'warning:' "$TMP/$name.err" >&2
    exit 1
  fi
  echo "OK: $name is offloaded without a warning"
}

check_offloaded_quietly int_matmul 1
check_offloaded_quietly conv_accel 1

# Input the compiler does not support is refused up front, with a message that
# says what is wrong. Arguments: input name, text the error must contain.
check_rejected() {
  local name="$1" message="$2"
  if "$ROOT/icdecompiler" "$ROOT/tests/inputs/$name.mlir" -o "$TMP/$name.c" \
       > /dev/null 2> "$TMP/$name.err"; then
    echo "FAIL: $name: compiled, expected an error" >&2
    exit 1
  fi
  if ! grep -q "$message" "$TMP/$name.err"; then
    echo "FAIL: $name: no error containing '$message'" >&2
    exit 1
  fi
  echo "OK: $name is rejected with a clear message"
}

check_rejected reject_dynamic_shape  "has a dynamic shape"
check_rejected reject_unranked       "has no rank"
check_rejected reject_element_type   "element type that is not supported"
check_rejected reject_call           "calls between functions are not supported"
check_rejected reject_rank0_constant "rank-0 constant is not supported"
check_rejected reject_no_function    "no function to compile"

# The generated C has to declare every function it calls: gcc 14 and later
# refuse a call to an undeclared function. Arguments: input name, mode.
check_declares_calls() {
  local name="$1" mode="${2:-}"
  "$ROOT/icdecompiler" "$ROOT/tests/inputs/$name.mlir" $mode -o "$TMP/$name.c" \
    > /dev/null 2>&1
  if ! gcc -fsyntax-only -Werror=implicit-function-declaration \
       "$TMP/$name.c" -I"$ROOT" -I"$HERE/stub-include" 2> "$TMP/$name.err"; then
    echo "FAIL: $name: generated C calls a function it does not declare" >&2
    grep 'implicit declaration' "$TMP/$name.err" >&2 || true
    exit 1
  fi
  echo "OK: $name declares every function it calls"
}

# a copy that stays in the output becomes a memcpy
check_declares_calls copy_reads_dest --cpu

# The same input must give the same C every time, byte for byte, or a rebuild
# shows up as a change in the generated file. This input has buffers of two
# element types, so two arenas, whose numbering used to vary between runs.
"$ROOT/icdecompiler" "$ROOT/tests/inputs/conv_q88.mlir" -o "$TMP/same_0.c" \
  > /dev/null 2>&1
for run in 1 2 3 4 5 6 7 8 9; do
  "$ROOT/icdecompiler" "$ROOT/tests/inputs/conv_q88.mlir" -o "$TMP/same_$run.c" \
    > /dev/null 2>&1
  if ! cmp -s "$TMP/same_0.c" "$TMP/same_$run.c"; then
    echo "FAIL: conv_q88: two compilations of the same input differ" >&2
    diff "$TMP/same_0.c" "$TMP/same_$run.c" | head -4 >&2
    exit 1
  fi
done
echo "OK: conv_q88 compiles to the same C every time"
