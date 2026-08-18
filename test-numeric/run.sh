#!/usr/bin/env bash
# Verify generated C produces numerically correct results.
# Uses a host-side stub driver that computes the same matmul in plain C.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

INPUT="${1:-$ROOT/test-input/int_matmul.mlir}"

"$ROOT/icdecompiler" "$INPUT" -o "$TMP/body.c" > /dev/null

{
  echo '#include <stdint.h>'
  echo '#include <stddef.h>'
  echo '#include "systolic_stub.h"'
  cat "$TMP/body.c"
} > "$TMP/gen.c"

gcc "$TMP/gen.c" "$HERE/main.c" -I"$HERE" -o "$TMP/numtest"
"$TMP/numtest"