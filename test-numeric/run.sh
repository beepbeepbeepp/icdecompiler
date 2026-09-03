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

  "$ROOT/icdecompiler" "$mlir" $mode -o "$TMP/body.c" > /dev/null

  {
    echo '#include <stdint.h>'
    echo '#include <stddef.h>'
    echo '#include "systolic_stub.h"'
    cat "$TMP/body.c"
  } > "$TMP/gen.c"

  gcc "$TMP/gen.c" "$HERE/$driver" -I"$HERE" -o "$TMP/test_$name"
  "$TMP/test_$name"
}

run_case "$ROOT/test-input/int_matmul.mlir"       main.c       square
run_case "$ROOT/test-input/int_matmul_rect.mlir"  main_rect.c  rect
run_case "$ROOT/test-input/tosa_matmul_i16.mlir"  main_batch.c batch
run_case "$ROOT/test-input/int_matmul.mlir"       main.c       square_cpu --cpu
run_case "$ROOT/test-input/q88_requant.mlir"      main_q88.c   q88