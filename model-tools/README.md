# model-tools

Mengubah model CNN terlatih menjadi kode C yang berjalan di RISC-V, dengan atau
tanpa systolic array. Bekerja bersama ICDeCompiler, yang menangani struktur
model; tool ini menangani bobot dan kuantisasi.

## Persiapan

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

ICDeCompiler harus sudah dibangun. Kalau letaknya bukan di `../icdecompiler`,
set `ICDECOMPILER` ke path executable-nya.

## Cara pakai

Satu perintah, dari model sampai kode C:

```bash
# model INT8, jalan di CPU
./build_model.sh --int8 model_int8.tflite har -o out/

# model float ke Q8.8, matmul ke systolic array
./build_model.sh --q88 savedmodeldir har -o out/ --array
```

Menghasilkan empat berkas di `out/`:

| Berkas | Isi |
|---|---|
| `har.mlir` | struktur model dalam TOSA |
| `har.c` | kode inferensi |
| `har_weights.c` `.h` | bobot dan parameter kuantisasi |

Nama `har` menentukan awalan simbol di kode C dan nama fungsi inferensinya.

## Pilih yang mana

| | INT8 | Q8.8 |
|---|---|---|
| Masukan | `.tflite` terkuantisasi | SavedModel float |
| Bobot | 1 byte per nilai | 2 byte per nilai |
| Kuantisasi | scale per kanal, dikalibrasi | skala seragam |
| Systolic array | perlu konversi | langsung |

Kalau targetnya CPU saja, INT8 lebih hemat memori. Kalau mau memakai array,
Q8.8 yang diterima drivernya.

Untuk Q8.8, mulai dari model float — bukan dari `.tflite` INT8 — supaya tidak
ada pembulatan dua kali.

## Yang didukung

Layer yang bisa diproses:

- Conv1D dan Conv2D
- MaxPooling
- Dense
- GlobalAveragePooling
- Reshape, Flatten
- Softmax (dilewati; argmax sudah cukup untuk klasifikasi)

Belum didukung: DepthwiseConv, koneksi residual, LSTM, GRU, AveragePooling,
Transpose.

Kalau model memakai layer yang belum didukung, importer akan menolak dan
menyebutkan layer mana. Menambahkannya berarti menulis satu fungsi di
`tools/tflite_to_tosa.py` atau `tools/keras_to_tosa_q88.py` dan
mendaftarkannya di `HANDLERS`.

## Memastikan hasilnya benar

Setelah kode C dihasilkan, bandingkan dengan model aslinya.

Butuh program pembungkus yang membaca masukan dari stdin dan mencetak keluaran.
`testbench/` menyediakan contohnya untuk model HAR.

```bash
# INT8, dibandingkan dengan interpreter TFLite
gcc -O2 -Iout testbench/test_host.c out/har.c out/har_weights.c -o har_test
python3 tools/verify_c.py model_int8.tflite ./har_test \
    --input-shape 512 3 --output-tensor 35

# Q8.8, dibandingkan dengan model float
gcc -O2 -Iout testbench/test_host_q88.c out/har.c out/har_weights.c -o har_q88_test
python3 tools/verify_q88.py savedmodeldir ./har_q88_test
```

Yang dilaporkan: berapa kali hasilnya sama persis, berapa kali kelas
prediksinya sama, dan selisih terbesar.

Kelas prediksi yang selalu sama sudah cukup untuk dipakai. Sama persis sampai
angka terakhir baru perlu kalau mau dipublikasikan.

## Struktur

model-tools/
├── build_model.sh satu perintah untuk seluruh alur
├── requirements.txt
├── tools/ script yang dipanggil build_model.sh
├── testbench/ pembungkus untuk verifikasi di PC
└── examples/har/ model contoh


## Hal yang perlu diketahui

Bobot layer dense di TFLite dan Keras tersimpan terbalik dari yang dibutuhkan
MLIR. Script sudah membalikkannya, jadi kode C harus mengikuti susunan yang
dihasilkan.

Untuk Q8.8, BatchNorm dilipat ke konvolusi sebelumnya. Bobot dan aktivasi
memakai `int16_t` dengan delapan bit pecahan; bias memakai `int32_t` Q16.16
supaya bisa dijumlahkan langsung ke akumulator.

Q8.8 hanya menampung nilai sekitar −128 sampai 128. Script melaporkan kalau ada
bobot yang tersaturasi. Untuk aktivasi, rentangnya perlu diperiksa sendiri —
model HAR aman, tapi model dengan nilai lebih besar mungkin tidak.