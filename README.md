# EvolKV dengan Genetic Algorithm

Implementasi ulang paper [EvolKV: Evolutionary KV Cache Compression for LLM Inference](https://arxiv.org/abs/2509.08315).
Paper memakai CMA-ES sebagai optimizer. Di sini optimizer-nya diganti dengan Genetic Algorithm, sedangkan
fitness, optimasi per grup layer, dan budget completion tetap mengikuti paper.

Tujuannya mencari budget KV cache untuk tiap layer agar performa model tetap tinggi walau cache dipangkas.

## Yang dibutuhkan

- Python 3.12
- GPU NVIDIA dengan VRAM 8 GB atau lebih. Tanpa GPU kode tetap jalan di CPU, tetapi jauh lebih lambat.
- Ruang disk sekitar 7 GB: 5,6 GB untuk dependensi dan 1 GB untuk model
- Internet saat setup dan saat model pertama kali diunduh

## Setup

Jalankan dari folder proyek ini. Lewati dua langkah ini bila `.venv` sudah ada.

```bash
python3 -m venv .venv
```

```bash
.venv/bin/pip install -r requirements.txt
```

Pastikan semuanya berfungsi. Tes memakai model kecil buatan sendiri, jadi tidak perlu mengunduh apa pun.

```bash
.venv/bin/python -m unittest test_evolkv test_kv_cache
```

## Menjalankan

**1. Cari budget dengan GA.** Pada GPU 8 GB butuh sekitar 8 menit. Model Qwen2.5-0.5B (sekitar 1 GB) diunduh
otomatis saat pertama kali dijalankan.

```bash
.venv/bin/python run.py --generations 8
```

**2. Buat grafik.** Langkah ini hanya membaca hasil langkah 1, jadi tidak perlu GPU dan selesai dalam hitungan detik.

```bash
.venv/bin/python plot.py
```

Untuk sekadar mengecek alurnya berjalan, pakai run pendek ini. Hasilnya terlalu sedikit untuk dinilai.

```bash
.venv/bin/python run.py --generations 1 --population 4 --train-samples 3 --test-samples 6 --context-tokens 1500
```

Setelah model terunduh, tambahkan `HF_HUB_OFFLINE=1` di depan perintah untuk menjalankan tanpa internet.

## Hasil

| File | Isi |
|---|---|
| `history.csv` | Riwayat GA per generasi: rentang layer, fitness terbaik, rata-rata populasi, kromosom terbaik |
| `results.json` | Recall Full KV, SnapKV, PyramidKV, dan EvolKV-GA, budget per layer, serta argumen run |
| `figures/convergence.png` | Fitness per generasi, dipisah per grup layer |
| `figures/budgets.png` | Alokasi budget per layer pada target utama |
| `figures/recall.png` | Recall data uji per target budget |

Hasil yang sudah ada (seed 0):

| c | SnapKV | PyramidKV | EvolKV-GA |
|---|---|---|---|
| 128 | 0,000 | 0,000 | 0,111 |
| 256 | 0,089 | 0,000 | 0,378 |
| 512 | 0,367 | 0,156 | 0,567 |

Recall Full KV adalah 0,856. Angka ini berasal dari data sintetis dan satu seed, jadi tidak bisa dibandingkan
langsung dengan tabel di paper.

## Argumen `run.py`

| Argumen | Default | Arti |
|---|---|---|
| `--model` | `Qwen/Qwen2.5-0.5B-Instruct` | Model Hugging Face |
| `--target` | 128 | Target rata-rata budget per layer (c) |
| `--expand` | 256 512 | Target tambahan, dihitung dengan menskalakan hasil GA |
| `--group-size` | 6 | Jumlah layer per grup (n_g) |
| `--generations` | 6 | Jumlah generasi GA per grup |
| `--population` | rumus paper | Ukuran populasi GA; defaultnya `4 + floor(3 * ln(n_g))` |
| `--high` | 4 × target | Batas atas nilai gen |
| `--window` | 32 | Jumlah token terakhir yang selalu disimpan, sekaligus batas bawah gen |
| `--kernel` | 7 | Ukuran kernel max pooling |
| `--context-tokens` | 3000 | Panjang konteks NIAH |
| `--needles` | 3 | Jumlah needle per sampel |
| `--train-samples` | 10 | Jumlah sampel untuk fitness |
| `--test-samples` | 30 | Jumlah sampel untuk evaluasi akhir |
| `--max-new-tokens` | 40 | Panjang jawaban model |
| `--seed` | 0 | Seed data dan GA |
| `--output` | `results.json` | File hasil |
| `--history` | `history.csv` | File riwayat GA |

Hasil di atas memakai `--generations 8`. Laju crossover, mutasi, ukuran turnamen, dan elitisme diatur di
`GAConfig` pada `ga.py`.

`plot.py` menerima `--history`, `--results`, dan `--output-dir` bila file-nya bukan di lokasi default.

## Struktur kode

| File | Isi |
|---|---|
| `ga.py` | Genetic Algorithm: seleksi turnamen, uniform crossover, Gaussian mutation, elitisme |
| `search.py` | Persamaan 1–2 paper, budget completion, alokasi baseline, dan Algoritma 1 |
| `kv_cache.py` | Prefill dan kompresi KV cache ala SnapKV dengan budget per layer |
| `niah.py` | Pembuat data Needle-in-a-Haystack dan skor recall |
| `run.py` | Menjalankan pencarian dan evaluasi akhir |
| `plot.py` | Membuat grafik dari `history.csv` dan `results.json` |
| `test_evolkv.py`, `test_kv_cache.py` | Tes unit |

## Batasan

- Pada c=128 dan konteks panjang, grup layer awal mendapat fitness 0 sehingga GA tidak punya sinyal. Pada hasil
  di atas, layer 0–5 tetap di budget awal.
- Baseline StreamingLLM belum tersedia, dan data uji hanya NIAH sintetis, bukan LongBench atau RULER.
- `kv_cache.py` mengikuti API internal transformers 5.22.1, jadi versinya dikunci di `requirements.txt`.
  Hanya model dengan struktur `model.layers` seperti Qwen2 yang sudah diuji.
