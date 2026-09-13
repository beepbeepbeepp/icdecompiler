# Contoh: model HAR

Model dan bobotnya tidak disertakan di repo karena merupakan hasil penelitian
lab. Yang ada di sini hanya TOSA hasil importer, sebagai acuan bentuk keluaran.

Untuk menjalankan contoh ini, salin dua hal ke folder ini:

| Berkas | Dipakai untuk |
|---|---|
| `model_int8.tflite` | jalur INT8 |
| `savedmodelhar/` | jalur Q8.8 |

Keduanya ada di `TinyML-ARM/har-ppg-imu-edge/experiments/20260708_acconly_deploy`.

Lalu:

```bash
./build_model.sh --int8 examples/har/model_int8.tflite har -o out/
./build_model.sh --q88  examples/har/savedmodelhar    har -o out/ --array
```
