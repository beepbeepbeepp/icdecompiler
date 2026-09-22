# ICDeCompiler

Compiles ML models to C for RISC-V targets, offloading matrix multiplication to
a systolic array accelerator when one is available.

Input is MLIR in the TOSA or linalg dialect; output is plain C that compiles
with GCC. There is no LLVM IR step, because the targets here use the PULP GCC
toolchain rather than clang.

## Building

Needs MLIR built from source. Point CMake at it:

```sh
mkdir build && cd build
cmake -G Ninja .. \
  -DMLIR_DIR=$LLVM_BUILD/lib/cmake/mlir \
  -DLLVM_DIR=$LLVM_BUILD/lib/cmake/llvm \
  -DCMAKE_BUILD_TYPE=Release
ninja
```

Built against LLVM 24. Earlier versions will not work: TOSA 1.0 renamed several
attributes, and the emitc conversion behaves differently.

On a machine with limited memory, use `ninja -j1`.

If you only want to use the compiler, not modify it, take a binary release
instead and skip building LLVM entirely.

## Using it

```sh
./icdecompiler model.mlir --cpu -o model.c    # everything on the CPU
./icdecompiler model.mlir -o model.c          # matmul goes to the array
```

In array mode the generated code calls `systolic_multiply_large` and includes
`systolic_drv.h`, so that header must sit next to the generated source when it
is compiled.

Weights are function parameters, not constants, so the MLIR stays small. The
matching weight arrays come from `model-tools`.

Every compilation prints a summary to stderr:

```
ICDeCompiler summary
  input             : 1x1x512x3 i16
  output            : 1x1x4 i16
  MACs total        : 805275
  on accelerator    : 612480  (76%) in 5 calls
  on CPU            : 192795  (23%)
  scratch memory    : 36864 bytes
```

Only multiply-accumulates are counted. Pooling, clamping and requantisation run
on the CPU too, but they are element-wise and cheap next to the matrix work, so
counting them the same way would misrepresent where the time goes.

Scratch memory is the working buffers, which the arena allocator packs by
lifetime. Weights are not included, since they arrive as parameters.

## Math functions on cores without an FPU

Operations like `exp` and `sigmoid` lower to libm calls such as `expf`. The
libm shipped with the PULP toolchain is built with the F extension, so on a
core without an FPU those calls trap.

For such targets, add `runtime/icdec_expf.c` to the firmware build. Compiled
with the application's flags it uses soft-float only, and the linker picks it
over the libm version. On a core that does have an FPU, leave it out: the libm
version is faster there.

Only models that actually use these operations need it. `grep expf model.c`
tells you.

## What it does

The pipeline lowers TOSA to linalg, bufferizes, then splits the work:
convolutions and matrix multiplies go to the accelerator, everything else
becomes loops. Convolutions reach the accelerator through im2col, which turns
them into matrix multiplies first.

Passes written for this project:

| Pass | What it does |
|---|---|
| `icdec-compile` | full pipeline, matmul offloaded |
| `icdec-compile-cpu` | full pipeline, CPU only |
| `icdec-conv-to-im2col` | rewrites convolutions as matrix multiplies |
| `icdec-matmul-to-systolic` | matmul becomes a driver call |
| `icdec-out-param` | memref returns become output parameters |
| `icdec-elide-copy-alloc` | drops temporaries that are only copied out |
| `icdec-alloc-to-arena` | packs buffers by lifetime into shared arenas |
| `icdec-global-to-emitc` | converts memref globals the emitc pass leaves behind |
| `icdec-expand-int-minmax` | integer min and max become compare plus select |
| `icdec-add-includes` | prepends the headers the output needs |
| `icdec-print-summary` | reports the MAC split and memory use |

Several of these exist because the bundled emitc conversion does not handle
those cases. They are not optimizations; without them the output does not
compile.

`icdec-alloc-to-arena` is the exception: it computes when each buffer is live
and lets buffers whose lifetimes do not overlap share memory. On the HAR model
this brings working memory down from about 70 KB to 24 KB, which is the
difference between fitting on the board and not.

## Testing

```sh
./test-numeric/run.sh     # generated code produces correct numbers
./test-compile/run.sh     # generated code compiles against the driver
```

`test-numeric` runs each case against a host-side stub that computes the same
matmul in plain C, and compares element by element. One case compiles both the
CPU and accelerator paths into the same program and checks they agree, which is
what catches divergence between the two.

## Adding an operator

Most models fail because an operator is missing rather than because the
pipeline is wrong. The importer in `model-tools` will say which one.

If TOSA already has the operator, the conversion to linalg usually exists
upstream and nothing needs to be added here. If it lowers to something the
matmul pass should catch, check `GenericMatmulToSystolicRewriter` in
`lib/Standalone/StandalonePasses.cpp`: it reads the indexing maps to work out
which dimensions are M, K and N, and rejects layouts it does not recognise
rather than producing wrong code.

## Known limits

No fusion, no tiling, no vectorization. The generated code is correct, not
fast. Quantization is the one thing that does make a large difference on these
targets, and that is handled by choosing INT8 or Q8.8 in `model-tools`.

Convolutions reach the accelerator through im2col, which materializes the
input windows as a matrix. That costs memory: a 512-long input with a 15-tap
kernel becomes a 256×45 buffer.

The driver limits K to 256, so layers with a larger reduction dimension will
not fit in one call.

Buffers are shared only when their element types match. A single byte arena
carved up by offset would pack tighter, but emitc has no way to express a
pointer into the middle of one.

## Layout

```
include/Standalone/   pass and dialect declarations
lib/Standalone/       pass implementations
standalone-opt/       the opt-like tool
standalone-translate/ MLIR to C
test-input/           MLIR test cases
test-numeric/         numeric verification
test-compile/         compile verification
icdecompiler          the script you actually run
make-release.sh       packages the built tools for distribution
```

The `standalone` naming comes from the MLIR out-of-tree template this started
from.