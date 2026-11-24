# Copyright 2026 Indonesia Chip Design Collaborative Center (ICDeC)
# Author: Muhammad Faudzan Abdullah
# SPDX-License-Identifier: Apache-2.0
#
# Flags the tests use to compile generated C: it has to build without a single
# warning. Sourced by the run.sh scripts, so the rule lives in one place.

STRICT_CFLAGS="-Wall -Werror"
