#!/usr/bin/env bash
# Verify generated C compiles against the real systolic driver header.
# Uses a stub pulp.h so this can run on the host without the PULP SDK.
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
  echo '#include "systolic_drv.h"'
  cat "$TMP/body.c"
} > "$TMP/full.c"

gcc -c "$TMP/full.c" -I"$ROOT" -I"$HERE/stub-include" -o "$TMP/out.o"

echo "OK: generated C compiles and links against systolic_drv.h"
nm "$TMP/out.o" | grep -q systolic_multiply_large \
  && echo "OK: systolic_multiply_large present in object"