# model-tools

Turns a trained CNN into C code that runs on RISC-V, with or without a systolic
array. Works alongside ICDeCompiler, which handles the model structure; this
handles weights and quantization.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

ICDeCompiler must already be built. If it is not at `../icdecompiler`, set
`ICDECOMPILER` to the path of its executable.

## Usage

One command, from model to C:

```bash
# INT8 model, runs on the CPU
./build_model.sh --int8 model_int8.tflite har -o out/

# float model to Q8.8, matmul goes to the systolic array
./build_model.sh --q88 savedmodeldir har -o out/ --array
```

Produces four files in `out/`:

| File | Contents |
|---|---|
| `har.mlir` | model structure as TOSA |
| `har.c` | inference code |
| `har_weights.c` `.h` | weights and quantization parameters |

The name `har` sets the symbol prefix in the C code and the name of the
inference function.

## Which one to use

| | INT8 | Q8.8 |
|---|---|---|
| Input | quantized `.tflite` | float SavedModel |
| Weights | 1 byte per value | 2 bytes per value |
| Quantization | per-channel scale, calibrated | uniform scale |
| Systolic array | needs conversion | direct |

For a CPU-only target, INT8 uses less memory. For the array, Q8.8 is what the
driver takes.

Start Q8.8 from the float model rather than an INT8 `.tflite`, so the values
are rounded once instead of twice.

## Supported layers

- Conv1D and Conv2D
- MaxPooling
- Dense
- GlobalAveragePooling
- Reshape, Flatten
- Softmax, which is skipped since argmax is enough for classification

Not yet supported: DepthwiseConv, residual connections, LSTM, GRU,
AveragePooling, Transpose.

A model using an unsupported layer is rejected by name. Adding one means
writing a function in `tools/tflite_to_tosa.py` or `tools/keras_to_tosa_q88.py`
and registering it in `HANDLERS`.

## Checking the output

Compare the generated C against the model it came from.

This needs a host wrapper that reads the input from stdin and prints the
output. `testbench/` has one for the HAR model.

```bash
# INT8, against the TFLite interpreter
gcc -O2 -Iout testbench/test_host.c out/har.c out/har_weights.c -o har_test
python3 tools/verify_c.py model_int8.tflite ./har_test \
    --input-shape 512 3 --output-tensor 35

# Q8.8, against the float model
gcc -O2 -Iout testbench/test_host_q88.c out/har.c out/har_weights.c -o har_q88_test
python3 tools/verify_q88.py savedmodeldir ./har_q88_test
```

Reports three things: how often the output matches exactly, how often the
predicted class matches, and the largest difference.

Matching classes every time is enough to use the model. Exact matches down to
the last digit only matter for publication.

## Layout

```
model-tools/
├── build_model.sh       one command for the whole flow
├── requirements.txt
├── tools/               scripts build_model.sh calls
├── testbench/           host wrappers for verification
└── examples/har/        example model
```

## Things worth knowing

Dense weights in TFLite and Keras are stored transposed from what MLIR needs.
The scripts transpose them, so the C code has to follow the layout they
produce.

For Q8.8, BatchNorm is folded into the preceding convolution. Weights and
activations use `int16_t` with eight fractional bits; biases use `int32_t` in
Q16.16 so they add straight into the accumulator.

Q8.8 only holds values from about −128 to 128. The script reports any weight
that saturates. Activation ranges are not checked automatically: the HAR model
stays well inside the range, but a model with larger values might not.