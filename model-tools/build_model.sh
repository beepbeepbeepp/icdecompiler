#!/usr/bin/env bash
# Turn a trained model into C source ready to compile for the target.
#
#   ./build_model.sh --int8 model_int8.tflite har -o out/
#   ./build_model.sh --q88  savedmodeldir     har -o out/ --array
#
# INT8 starts from a quantized .tflite and runs on the CPU.
# Q8.8 starts from a float SavedModel and is what the systolic array expects.
#
# Produces, in the output directory:
#   <name>.mlir          model structure as TOSA
#   <name>.c             inference code
#   <name>_weights.c/.h  weights and quantization parameters

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLS="$HERE/tools"
COMPILER="${ICDECOMPILER:-$HERE/../icdecompiler/icdecompiler}"

usage() {
    cat >&2 <<'EOF'
usage: build_model.sh --int8|--q88 <model> <name> [-o outdir] [--array]

  --int8    quantized .tflite input, INT8 weights
  --q88     float SavedModel input, Q8.8 weights
  --array   offload matmul to the systolic array (default: CPU only)
EOF
    exit 1
}

[[ $# -lt 3 ]] && usage

FORMAT="$1"; shift
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

case "$FORMAT" in
    --int8) EXTRACT="extract_weights.py";     IMPORT="tflite_to_tosa.py" ;;
    --q88)  EXTRACT="extract_weights_q88.py"; IMPORT="keras_to_tosa_q88.py" ;;
    *)      echo "unknown format: $FORMAT" >&2; usage ;;
esac

[[ -e "$MODEL" ]] || { echo "no such model: $MODEL" >&2; exit 1; }
[[ -x "$COMPILER" ]] || { echo "compiler not found: $COMPILER" >&2; exit 1; }

mkdir -p "$OUTDIR"

echo "==> extracting weights"
python3 "$TOOLS/$EXTRACT" "$MODEL" -o "$OUTDIR/${NAME}_weights" --prefix "$NAME"

echo
echo "==> importing model structure to TOSA"
python3 "$TOOLS/$IMPORT" "$MODEL" -o "$OUTDIR/${NAME}.mlir" --name "$NAME"

echo
echo "==> compiling to C"
"$COMPILER" "$OUTDIR/${NAME}.mlir" $MODE -o "$OUTDIR/${NAME}.c"

echo
echo "done. generated in $OUTDIR:"
ls -1 "$OUTDIR/${NAME}".* "$OUTDIR/${NAME}_weights".* 2>/dev/null | sed 's/^/  /'