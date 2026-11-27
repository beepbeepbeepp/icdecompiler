#!/usr/bin/env bash
# Copyright 2026 Indonesia Chip Design Collaborative Center (ICDeC)
# Author: Muhammad Faudzan Abdullah
# SPDX-License-Identifier: Apache-2.0
#
# Run every test suite. Exits non-zero on the first failure.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "== numeric =="
"$HERE/numeric/run.sh"
echo
echo "== compile =="
"$HERE/compile/run.sh"
echo
echo "== operator coverage =="
"$HERE/ops/run.sh"

echo
echo "== unused inputs =="
# Every file in tests/inputs has to be used by a test. One that is not is
# either a case nobody checks or a leftover, and both mislead whoever reads
# the directory. A name that only appears in a comment does not count.
used="$(grep -h -v '^[[:space:]]*#' "$HERE/numeric/run.sh" "$HERE/compile/run.sh")"
unused=()
for input in "$HERE"/inputs/*; do
  name="$(basename "$input" .mlir)"
  grep -qE "(inputs/|[[:space:]])$name(\.mlir|[[:space:]]|\$)" <<< "$used" \
    || unused+=("$name")
done
if [[ ${#unused[@]} -gt 0 ]]; then
  echo "FAIL: ${#unused[@]} files in tests/inputs are not used by any test: ${unused[*]}" >&2
  exit 1
fi
echo "OK: every file in tests/inputs is used by a test"
