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

# K limit: an op wider than the accelerator takes stays on the CPU with a
# warning, and the rest of the model still compiles.
check_k_limit() {
  local name="$1" want_calls="$2" calls
  "$ROOT/icdecompiler" "$ROOT/tests/inputs/$name.mlir" -o "$TMP/$name.c" \
    > /dev/null 2> "$TMP/$name.err"
  calls=$(grep -c systolic_multiply_large "$TMP/$name.c" || true)
  if [[ "$calls" != "$want_calls" ]]; then
    echo "FAIL: $name: $calls accelerator calls, expected $want_calls" >&2
    exit 1
  fi
  if ! grep -q "exceeds the accelerator limit" "$TMP/$name.err"; then
    echo "FAIL: $name: no K limit warning" >&2
    exit 1
  fi
  if grep -q 'malloc\|free' "$TMP/$name.c"; then
    echo "FAIL: $name: generated C uses malloc or free" >&2
    exit 1
  fi
  echo "OK: $name keeps the wide op on the CPU and warns"
}

check_k_limit matmul_k_limit 1
check_k_limit conv_k_limit 0