#!/usr/bin/env bash
# Verify generated C produces numerically correct results.
# Uses a host-side stub driver that computes the same matmul in plain C.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

run_case() {
  local mlir="$1"
  local driver="$2"
  local name="$3"
  local mode="${4:-}"

  "$ROOT/icdecompiler" "$mlir" $mode -o "$TMP/body.c" > /dev/null 2>&1

  {
    echo '#include <stdint.h>'
    echo '#include <stddef.h>'
    echo '#include "systolic_stub.h"'
    cat "$TMP/body.c"
  } > "$TMP/gen.c"

  gcc "$TMP/gen.c" "$HERE/$driver" -I"$HERE" ${CASE_CFLAGS:-} -o "$TMP/test_$name" -lm
  "$TMP/test_$name"
}

run_case "$ROOT/test-input/int_matmul.mlir"       main.c       square
run_case "$ROOT/test-input/int_matmul_rect.mlir"  main_rect.c  rect
run_case "$ROOT/test-input/tosa_matmul_i16.mlir"  main_batch.c batch
run_case "$ROOT/test-input/int_matmul.mlir"       main.c       square_cpu --cpu
run_case "$ROOT/test-input/q88_requant.mlir"      main_q88.c   q88
run_case "$ROOT/test-input/rescale.mlir"          main_rescale.c rescale --cpu
run_case "$ROOT/test-ops/reduce_max.mlir"         main_reduce_max.c reduce_max --cpu
run_case "$ROOT/test-ops/exp.mlir"                main_exp.c   exp --cpu
CASE_CFLAGS=-DSIGMOID run_case "$ROOT/test-ops/sigmoid.mlir" main_exp.c sigmoid --cpu

# The accelerator path must produce the same numbers as the CPU path, so both
# are compiled into one program and compared element by element.
"$ROOT/icdecompiler" "$ROOT/test-input/conv_accel.mlir" --cpu -o "$TMP/conv_cpu.c" > /dev/null 2>&1
"$ROOT/icdecompiler" "$ROOT/test-input/conv_accel.mlir"        -o "$TMP/conv_acc.c" > /dev/null 2>&1

sed 's/conv_accel/conv_cpu/' "$TMP/conv_cpu.c" > "$TMP/cc1.c"
{
  echo '#include "systolic_stub.h"'
  sed 's/conv_accel/conv_acc/' "$TMP/conv_acc.c"
} > "$TMP/cc2.c"

gcc -O2 -I"$HERE" "$TMP/cc1.c" "$TMP/cc2.c" "$HERE/main_conv_accel.c" \
    -o "$TMP/test_conv_accel"
"$TMP/test_conv_accel"