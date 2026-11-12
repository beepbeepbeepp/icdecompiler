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
