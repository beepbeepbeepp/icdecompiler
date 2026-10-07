# ICDeCompiler

Compiles ML models to C for RISC-V targets, offloading matrix multiplication to
a systolic array accelerator when one is available.

A trained model goes in; C code that GCC compiles for the target comes out.
Input to the compiler proper is MLIR in the TOSA dialect, produced from TFLite
or Keras models by the tools in `model-tools/`. There is no LLVM IR step,
because the targets here use the PULP GCC toolchain rather than clang.

## Quick start

Using a binary release, no LLVM build needed:

```sh
tar xzf icdecompiler-bin-*.tar.gz && cd icdecompiler
python3 -m venv model-tools/.venv
source model-tools/.venv/bin/activate
pip install -r model-tools/requirements.txt

./model-tools/build_model.sh --int8 model.tflite net -o out/
```

That produces `net.c`, `net_weights.c`, `net_weights.h` and `net.mlir` in
`out/`. Add them to the firmware next to `main.c` and build with the target's
toolchain.

For the accelerator, start from the float model and quantize to Q8.8:

```sh
./model-tools/build_model.sh --q88 savedmodel net -o out/ --array
```

Every compilation prints a summary:

```
ICDeCompiler summary
  input             : 1x1x512x3 i16
  output            : 1x1x4 i16
  MACs total        : 612480
  on accelerator    : 612480  (100%) in 5 calls
  on CPU            : 0  (0%)
  scratch memory    : 60398 bytes
```

Only multiply-accumulates on model data are counted; index arithmetic and
requantisation are left out. Scratch memory is the working buffers; weights are
not included, since they arrive as parameters.

## What is in this repository

```
icdecompiler          compiler driver, the script you run
model-tools/          weight extraction, TFLite/Keras to TOSA, verification
runtime/              soft-float expf for cores without an FPU
tests/                numeric, compile and operator-coverage suites
docs/                 guides
make-release.sh       packages compiler and tools for distribution
include/ lib/         compiler passes
standalone-opt/       MLIR pass pipeline tool
standalone-translate/ MLIR to C
```

The `standalone` naming comes from the MLIR out-of-tree template this started
from.

## INT8 or Q8.8

| | INT8 | Q8.8 |
|---|---|---|
| Starts from | quantized `.tflite` | float SavedModel |
| Weights | 1 byte per value | 2 bytes per value |
| Accelerator | not yet, runs on CPU | yes |

INT8 layers carry zero points, and the accelerator path only takes Q8.8 for
now. With `--array` an INT8 model still falls back to the CPU and stays
correct.

## Building from source

Needs MLIR built from source, LLVM 24. Earlier versions will not work: TOSA
1.0 renamed several attributes, and the emitc conversion behaves differently.

```sh
mkdir build && cd build
cmake -G Ninja .. \
  -DMLIR_DIR=$LLVM_BUILD/lib/cmake/mlir \
  -DLLVM_DIR=$LLVM_BUILD/lib/cmake/llvm \
  -DCMAKE_BUILD_TYPE=Release
ninja
```

On a machine with limited memory, use `ninja -j1`.

## Testing

```sh
./tests/run-all.sh
```

Runs three suites. `numeric` checks generated code against reference results,
including one case that compiles the CPU and accelerator paths into the same
program and checks they agree. `compile` checks the output builds against the
driver header. `ops` reports which TOSA operators reach C and which of them go
to the accelerator.

## How it works

The pipeline lowers TOSA to linalg, bufferizes, then splits the work:
convolutions and matrix multiplies go to the accelerator, everything else
becomes loops. Convolutions reach the accelerator through im2col, which turns
them into matrix multiplies first.

| Pass | What it does |
|---|---|
| `icdec-compile` | full pipeline, matmul offloaded |
| `icdec-compile-cpu` | full pipeline, CPU only |
| `icdec-conv-to-im2col` | rewrites convolutions as matrix multiplies |
| `icdec-matmul-to-systolic` | matmul becomes a driver call |
| `icdec-out-param` | memref returns become output parameters |
| `icdec-elide-copy-alloc` | drops temporaries that are only copied out |
| `icdec-reinterpret-cast-to-index` | accesses through a reshape address the source directly |
| `icdec-alloc-to-arena` | packs buffers by lifetime into shared arenas |
| `icdec-global-to-emitc` | converts memref globals the emitc pass leaves behind |
| `icdec-expand-int-minmax` | integer min and max become compare plus select |
| `icdec-add-includes` | prepends the headers the output needs |
| `icdec-print-summary` | reports the MAC split and memory use |

Most of these exist because the bundled emitc conversion does not handle those
cases; without them the output does not compile. `icdec-alloc-to-arena` is the
exception: on the HAR model it brings working memory from about 70 KB to 24 KB,
which is the difference between fitting on the board and not.

## Math functions on cores without an FPU

`exp` and `sigmoid` lower to libm calls such as `expf`. The libm shipped with
the PULP toolchain is built with the F extension, so on a core without an FPU
those calls trap. For such targets, add `runtime/icdec_expf.c` to the firmware
build; it uses soft-float only and the linker picks it over libm. On a core
with an FPU, leave it out.

Only models that use these operations need it. `grep expf net.c` tells you.

## Known limits

No fusion, no tiling, no vectorization. The generated code is correct, not
fast.

INT8 models do not reach the accelerator yet; see above.

The driver limits K to 256, so layers with a larger reduction dimension will
not fit in one accelerator call.

im2col materializes the input windows as a matrix, which costs memory: a
512-long input with a 15-tap kernel becomes a 256×45 buffer.

Buffers share memory only when their element types match.