#!/usr/bin/env bash
# Package the built compiler so it can be used without building LLVM.
#
#   ./make-release.sh
#
# Produces icdecompiler-bin-<date>.tar.gz. Upload it as a GitHub release; the
# install script on the other end unpacks it.

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

cat > "$DIST/README.md" <<'EOF'
# ICDeCompiler, binary release

Compiler siap pakai, tidak perlu membangun LLVM.

## Pasang

```bash
tar xzf icdecompiler-bin-*.tar.gz
sudo ln -s "$PWD/icdecompiler/icdecompiler" /usr/local/bin/icdecompiler
```

Atau tanpa symlink, panggil langsung dengan path lengkapnya.

## Pakai

```bash
icdecompiler model.mlir --cpu -o model.c    # semua di CPU
icdecompiler model.mlir -o model.c          # matmul ke systolic array
```

Untuk mode array, `systolic_drv.h` harus ada di folder yang sama dengan kode C
saat dikompilasi. Versi yang disertakan di sini adalah driver AXI lama; pakai
driver HWPE dari tim hardware kalau targetnya board dengan HWPE.

Butuh Ubuntu dengan glibc yang tidak terlalu jauh berbeda dari mesin pembuat.
EOF

tar czf "$OUT" -C "$STAGE" icdecompiler

echo "wrote $OUT"
ls -lh "$OUT"