# model-tools

Menyiapkan model TFLite terkuantisasi INT8 untuk dijalankan di target bare-metal:
mengekstrak bobot dan parameter kuantisasi menjadi kode C, lalu memverifikasi
implementasi C terhadap interpreter TFLite.

Dipakai di hulu ICDeCompiler. Compiler mengurus struktur model; tool ini
mengurus bobot dan parameter kuantisasi.

## Persiapan

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install tflite numpy ai-edge-litert
```
## Cara cepat

Satu perintah untuk menghasilkan semua berkas:

```bash
./build_model.sh model_int8.tflite har -o output/
```

Menghasilkan:

| Berkas | Isi |
|---|---|
| `har.mlir` | struktur model dalam TOSA |
| `har.c` | kode inferensi |
| `har_weights.c` `.h` | bobot dan parameter kuantisasi |

Tambahkan `--array` kalau matmul mau diarahkan ke systolic array. Tanpa itu,
semua dijalankan CPU.

Bagian di bawah menjelaskan tiap langkahnya kalau perlu dijalankan terpisah.

## Ekstraksi bobot

```bash
python3 extract_weights.py model_int8.tflite -o har_weights_int8 --prefix har
```

Menghasilkan tiga berkas:

| Berkas | Isi |
|---|---|
| `har_weights_int8.c` | bobot int8, bias int32, pengali requantisasi, shift |
| `har_weights_int8.h` | deklarasi extern dan makro zero point |
| `har_weights_int8.json` | ringkasan parameter tiap layer |

Opsi `--prefix` menentukan awalan nama simbol C. Tanpa itu, nama berkas keluaran
yang dipakai.

## Verifikasi implementasi C

Butuh program C yang membaca masukan terkuantisasi dari stdin sebagai bilangan
bulat dipisah spasi, lalu mencetak logits dengan cara yang sama.
`test_host.c` menyediakan pembungkus untuk model HAR.

```bash
gcc -O2 -I. test_host.c ../pulp-nusacore/app-har-int8/har_int8.c \
    har_weights_int8.c -o har_test

python3 verify_c.py model_int8.tflite ./har_test \
    --input-shape 512 3 --output-tensor 35 --trials 30
```

Keluarannya melaporkan tiga hal: berapa kasus yang bit-exact, berapa yang
kelasnya cocok, dan selisih absolut terbesar.

Opsi `--output-tensor` menunjuk tensor yang dibandingkan. Untuk model dengan
softmax di akhir, biasanya yang dibandingkan adalah keluaran layer sebelumnya,
karena implementasi C umumnya berhenti di logits.

## Catatan

Bobot dense pada TFLite tersimpan sebagai `[out, in]`, sedangkan `tosa.matmul`
mengharapkan `[1, in, out]`. Script sudah melakukan transpose ini; kode C yang
memakainya harus mengikuti tata letak tersebut.

Pengali requantisasi dihitung dengan port dari `QuantizeMultiplier` milik
TFLite, menghasilkan pasangan multiplier Q31 dan right shift per kanal keluaran.

Operasi MEAN pada TFLite tidak diekstrak oleh script ini karena tidak punya
bobot. Parameter kuantisasinya perlu diambil manual dari `.json` yang dihasilkan
atau langsung dari model.