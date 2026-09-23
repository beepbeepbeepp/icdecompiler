#!/usr/bin/env bash
# Package the built compiler and model tools so they can be used without
# building LLVM.
#
#   ./make-release.sh
#
# Produces icdecompiler-bin-<date>.tar.gz. Upload it as a GitHub release.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="$HERE/build/bin"
STAGE="$(mktemp -d)"
DIST="$STAGE/icdecompiler"
DATE=$(date +%Y%m%d)
OUT="$HERE/icdecompiler-bin-${DATE}.tar.gz"

trap 'rm -rf "$STAGE"' EXIT

for tool in standalone-opt standalone-translate; do
    [[ -x "$BUILD/$tool" ]] || { echo "missing $BUILD/$tool; build first" >&2; exit 1; }
done

mkdir -p "$DIST"

# Stripping drops debug symbols the tools do not need, roughly a quarter of the
# size.
for tool in standalone-opt standalone-translate; do
    cp "$BUILD/$tool" "$DIST/"
    strip "$DIST/$tool"
done

cp "$HERE/icdecompiler" "$DIST/"
cp "$HERE/systolic_drv.h" "$DIST/"

# Tracked files only: no venv, no caches, no lab models that are gitignored.
# Note this packages the last commit, not uncommitted edits.
git -C "$HERE" archive HEAD model-tools runtime | tar -x -C "$DIST"

cat > "$DIST/README.md" <<'EOF'
# ICDeCompiler, binary release

Compiler and model tools, ready to use without building LLVM.

## Contents

    icdecompiler            compiler driver
    standalone-opt          MLIR pass pipeline
    standalone-translate    MLIR to C
    model-tools/            weight extraction, importers, verification
    runtime/                soft-float expf for cores without an FPU
    systolic_drv.h          accelerator driver interface

## Setup

    python3 -m venv model-tools/.venv
    source model-tools/.venv/bin/activate
    pip install -r model-tools/requirements.txt

## Use

    ./model-tools/build_model.sh --int8 model.tflite net -o out/
    ./model-tools/build_model.sh --q88  savedmodel   net -o out/ --array

Produces net.c, net_weights.c, net_weights.h and net.mlir in out/.

In array mode the generated code includes systolic_drv.h. The copy here is the
interface the compiler targets; use the driver from the hardware team for the
actual board.

Needs a Linux with a glibc not much older than the machine that built this.
EOF

tar czf "$OUT" -C "$STAGE" icdecompiler

echo "wrote $OUT"
ls -lh "$OUT"