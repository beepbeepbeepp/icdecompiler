#!/usr/bin/env bash
# Turn a quantized TFLite model into C source ready to compile for the target.
#
#   ./build_model.sh model_int8.tflite har
#
# Produces, in the output directory:
#   <name>.mlir          model structure as TOSA
#   <name>.c             inference code
#   <name>_weights.c/.h  weights and quantization parameters

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPILER="${ICDECOMPILER:-$HERE/../icdecompiler/icdecompiler}"

usage() {
    echo "usage: build_model.sh <model.tflite> <name> [-o outdir] [--array]" >&2
    echo "  --array   offload matmul to the systolic array (default: CPU only)" >&2
    exit 1
}

[[ $# -lt 2 ]] && usage

MODEL="$1"; shift
NAME="$1"; shift
OUTDIR="."
MODE="--cpu"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -o)      OUTDIR="$2"; shift 2 ;;
        --array) MODE=""; shift ;;
        *)       echo "unknown option: $1" >&2; usage ;;
    esac
done

[[ -f "$MODEL" ]] || { echo "no such model: $MODEL" >&2; exit 1; }
[[ -x "$COMPILER" ]] || { echo "compiler not found: $COMPILER" >&2; exit 1; }

mkdir -p "$OUTDIR"

echo "==> extracting weights"
python3 "$HERE/extract_weights.py" "$MODEL" \
    -o "$OUTDIR/${NAME}_weights" --prefix "$NAME"

echo
echo "==> importing model structure to TOSA"
python3 "$HERE/tflite_to_tosa.py" "$MODEL" \
    -o "$OUTDIR/${NAME}.mlir" --name "$NAME"

echo
echo "==> compiling to C"
"$COMPILER" "$OUTDIR/${NAME}.mlir" $MODE -o "$OUTDIR/${NAME}.c"

echo
echo "done. generated in $OUTDIR:"
ls -1 "$OUTDIR/${NAME}".* "$OUTDIR/${NAME}_weights".* 2>/dev/null | sed 's/^/  /'