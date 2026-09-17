#!/usr/bin/env bash
# Work out which TOSA operators make it all the way to C, and where they land.
#
# For each case in this directory, compile twice: once CPU-only and once with
# the accelerator enabled. Report whether the C came out, and whether any of it
# went to the array.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

printf '%-18s %-8s %-10s %s\n' "operator" "to C" "on array" "note"
printf '%-18s %-8s %-10s %s\n' "------------------" "--------" "----------" "----"

for mlir in "$HERE"/*.mlir; do
    name=$(basename "$mlir" .mlir)

    # CPU path: does it reach C at all?
    if "$ROOT/icdecompiler" "$mlir" --cpu -o "$TMP/$name.c" > "$TMP/$name.log" 2>&1; then
        reached="yes"
    else
        reached="no"
        # first error line is usually the operation that could not be lowered
        note=$(grep -m1 "error:" "$TMP/$name.log" | sed 's/.*error: //' | cut -c1-48)
        printf '%-18s %-8s %-10s %s\n' "$name" "$reached" "-" "$note"
        continue
    fi

    # Array path: did any of it become a driver call?
    if "$ROOT/icdecompiler" "$mlir" -o "$TMP/${name}_acc.c" > /dev/null 2>&1; then
        calls=$(grep -c "systolic_multiply_large" "$TMP/${name}_acc.c")
        if [[ "$calls" -gt 0 ]]; then
            onarray="$calls calls"
        else
            onarray="no"
        fi
    else
        onarray="failed"
    fi

    printf '%-18s %-8s %-10s %s\n' "$name" "$reached" "$onarray" ""
done