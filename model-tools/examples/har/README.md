# Example: HAR model

The model and its weights are not in the repository, since they are lab
research output. What is here is the TOSA produced by the importer, as a
reference for what the output looks like.

To run this example, copy two things into this directory:

| File | Used for |
|---|---|
| `model_int8.tflite` | the INT8 path |
| `savedmodelhar/` | the Q8.8 path |

Both live in `TinyML-ARM/har-ppg-imu-edge/experiments/20260708_acconly_deploy`.

Then:

```bash
./build_model.sh --int8 examples/har/model_int8.tflite har -o out/
./build_model.sh --q88  examples/har/savedmodelhar    har -o out/ --array
```