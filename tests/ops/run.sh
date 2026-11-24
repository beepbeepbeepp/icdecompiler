#!/usr/bin/env bash
# Copyright 2026 Indonesia Chip Design Collaborative Center (ICDeC)
# Author: Muhammad Faudzan Abdullah
# SPDX-License-Identifier: Apache-2.0
#
# Work out which TOSA operators make it all the way to C, and where they land.
#
# For each case in this directory, compile twice: once CPU-only and once with
# the accelerator enabled. Report whether the C came out, and whether any of it
# went to the array.
#
# Every case here is an operator the compiler supports, so each one has to
# reach C on both paths, and that C has to be valid. The table is printed in
# full first; then the script fails if any case did not make it.
set -uo pipefail   # no -e: a case that fails is reported, it does not stop the table

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$(dirname "$HERE")")"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
source "$ROOT/tests/strict-cflags.sh"

# Is this file C that a compiler accepts? The accelerator path includes the
# driver header, which needs the stub pulp.h to be read on the host.
valid_c() {
    gcc $STRICT_CFLAGS -S "$1" -I"$ROOT" -I"$ROOT/tests/compile/stub-include" \
        -o /dev/null > "$TMP/gcc.log" 2>&1
}

first_error() {
    grep -m1 "error:" "$1" | sed 's/.*error: //' | cut -c1-48
}

total=0
failed=()

printf '%-18s %-8s %-10s %s\n' "operator" "to C" "on array" "note"
printf '%-18s %-8s %-10s %s\n' "------------------" "--------" "----------" "----"

for mlir in "$HERE"/*.mlir; do
    name=$(basename "$mlir" .mlir)
    total=$((total + 1))

    # CPU path: does it reach C at all?
    if ! "$ROOT/icdecompiler" "$mlir" --cpu -o "$TMP/$name.c" > "$TMP/$name.log" 2>&1; then
        # first error line is usually the operation that could not be lowered
        printf '%-18s %-8s %-10s %s\n' "$name" "no" "-" "$(first_error "$TMP/$name.log")"
        failed+=("$name")
        continue
    fi
    if ! valid_c "$TMP/$name.c"; then
        printf '%-18s %-8s %-10s %s\n' "$name" "bad C" "-" "$(first_error "$TMP/gcc.log")"
        failed+=("$name")
        continue
    fi

    # Array path: did any of it become a driver call?
    note=""
    if ! "$ROOT/icdecompiler" "$mlir" -o "$TMP/${name}_acc.c" > "$TMP/${name}_acc.log" 2>&1; then
        onarray="failed"
        note=$(first_error "$TMP/${name}_acc.log")
        failed+=("$name")
    elif ! valid_c "$TMP/${name}_acc.c"; then
        onarray="bad C"
        note=$(first_error "$TMP/gcc.log")
        failed+=("$name")
    else
        calls=$(grep -c "systolic_multiply_large_hwpe(" "$TMP/${name}_acc.c" || true)
        if [[ "$calls" -gt 0 ]]; then
            onarray="$calls calls"
        else
            onarray="no"
        fi
    fi

    printf '%-18s %-8s %-10s %s\n' "$name" "yes" "$onarray" "$note"
done

echo
if [[ ${#failed[@]} -gt 0 ]]; then
    echo "FAIL: ${#failed[@]} of $total operators do not compile to valid C: ${failed[*]}" >&2
    exit 1
fi
echo "OK: all $total operators compile to valid C on both paths"
