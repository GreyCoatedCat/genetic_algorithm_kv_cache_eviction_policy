# EvolKV dengan Genetic Algorithm

Implementasi ulang paper [EvolKV: Evolutionary KV Cache Compression for LLM Inference](https://arxiv.org/abs/2509.08315).
Paper memakai CMA-ES sebagai optimizer. Di sini optimizer-nya diganti dengan Genetic Algorithm, sedangkan
fitness, optimasi per grup layer, dan perluasan budget tetap mengikuti paper.

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

**1. Cari budget dengan GA.** Secara default dijalankan untuk seed 0, 1, dan 2, masing-masing sekitar 7,5 menit
pada GPU 8 GB. Model Qwen2.5-0.5B (sekitar 1 GB) diunduh otomatis saat pertama kali dijalankan.

```bash
.venv/bin/python run.py
```

**2. Buat grafik.** Langkah ini merangkum semua seed di `runs/`, mencetak tabel rata-rata, dan tidak perlu GPU.

```bash
.venv/bin/python plot.py
```

Untuk sekadar mengecek alurnya berjalan, pakai run pendek ini. Hasilnya ditulis ke folder terpisah agar
`runs/` tidak tertimpa, dan terlalu sedikit untuk dinilai.

```bash
.venv/bin/python run.py --seeds 0 --generations 1 --population 4 --train-samples 3 --test-samples 6 --context-tokens 1500 --output-dir runs_coba
```

```bash
.venv/bin/python plot.py --runs runs_coba --output-dir figures_coba
```

Setelah model terunduh, tambahkan `HF_HUB_OFFLINE=1` di depan perintah untuk menjalankan tanpa internet.

## Cara kerja

1. **Data:** dokumen sekitar 3000 token berisi 3 "needle" (angka 7 digit). Model diminta menyebutkan ketiganya.
2. **Kompresi:** setelah prefill, tiap layer hanya menyimpan token yang paling diperhatikan oleh 32 token terakhir
   (cara SnapKV), sebanyak budget layer itu.
3. **Kromosom:** budget untuk satu grup berisi 6 layer. GA mengoptimasi grup satu per satu dari layer bawah, sementara
   grup lain dibekukan (Algoritma 1 paper).
4. **Batas budget:** setiap kandidat diskalakan agar rata-rata budget tepat `c`. Jadi gen menyatakan proporsi antar-layer,
   dan mengurangi satu layer langsung memberi budget ke layer lain.
5. **Fitness:** Persamaan 1 paper, `f(S) * (1 + λ * CacheScore)`. Di sini `f(S)` adalah skor parsial per digit pada
   10 sampel latih. Skor ini dipakai karena pada budget kecil model sering benar di awal angka lalu mengarang. Dengan
   recall ketat semua kandidat bernilai 0, sehingga GA tidak punya sinyal.
6. **Evaluasi:** recall ketat pada 30 sampel uji yang berbeda, dibandingkan dengan SnapKV (budget sama rata) dan PyramidKV
   (rumus KVCache-Factory, menurun dari layer bawah). Budget c=128 juga diskalakan ke 256 dan 512 tanpa pencarian ulang.

## Hasil

| File | Isi |
|---|---|
| `runs/seed_<n>/history.csv` | Riwayat GA per generasi: rentang layer, fitness terbaik, rata-rata populasi, recall latih, kromosom terbaik |
| `runs/seed_<n>/results.json` | Recall Full KV, SnapKV, PyramidKV, dan EvolKV-GA, budget per layer, serta argumen run |
| `figures/convergence.png` | Fitness dan recall latih per generasi, rata-rata dan rentang antar seed |
| `figures/budgets.png` | Alokasi budget per layer pada c=128 |
| `figures/recall.png` | Recall data uji per target budget |

Hasil yang sudah ada, rata-rata recall uji dari 3 seed (Full KV 0,856):

| c | SnapKV | PyramidKV | EvolKV-GA | Rentang EvolKV-GA |
|---|---|---|---|---|
| 128 | 0,011 | 0,007 | 0,167 | 0,089–0,267 |
| 256 | 0,093 | 0,011 | 0,485 | 0,333–0,644 |
| 512 | 0,400 | 0,207 | 0,800 | 0,722–0,900 |

EvolKV-GA lebih tinggi dari kedua baseline pada setiap seed dan setiap target. Angka ini berasal dari data sintetis
dan model kecil, jadi tidak bisa dibandingkan langsung dengan tabel di paper.

## Argumen `run.py`

| Argumen | Default | Arti |
|---|---|---|
| `--model` | `Qwen/Qwen2.5-0.5B-Instruct` | Model Hugging Face |
| `--target` | 128 | Target rata-rata budget per layer (c) |
| `--expand` | 256 512 | Target tambahan, dihitung dengan menskalakan hasil GA |
| `--group-size` | 6 | Jumlah layer per grup (n_g) |
| `--generations` | 8 | Jumlah generasi GA per grup |
| `--population` | rumus paper | Ukuran populasi GA; defaultnya `4 + floor(3 * ln(n_g))` |
| `--high` | 4 × target | Batas atas nilai gen sebelum diskalakan |
| `--window` | 32 | Jumlah token terakhir yang selalu disimpan, sekaligus batas bawah budget |
| `--kernel` | 7 | Ukuran kernel max pooling |
| `--context-tokens` | 3000 | Panjang konteks NIAH |
| `--needles` | 3 | Jumlah needle per sampel |
| `--train-samples` | 10 | Jumlah sampel untuk fitness |
| `--test-samples` | 30 | Jumlah sampel untuk evaluasi akhir |
| `--max-new-tokens` | 40 | Panjang jawaban model |
| `--seeds` | 0 1 2 | Seed data dan GA; tiap seed dijalankan terpisah |
| `--output-dir` | `runs` | Folder hasil; tiap seed ditulis ke `<folder>/seed_<n>` |

Laju crossover, mutasi, ukuran turnamen, dan elitisme diatur di `GAConfig` pada `ga.py`.

`plot.py` menerima `--runs` (folder hasil, default `runs`) dan `--output-dir` (default `figures`).

## Struktur kode

| File | Isi |
|---|---|
| `ga.py` | Genetic Algorithm: seleksi turnamen, uniform crossover, Gaussian mutation, elitisme |
| `search.py` | Persamaan 1–2 paper, penskalaan budget, alokasi baseline, dan Algoritma 1 |
| `kv_cache.py` | Prefill dan kompresi KV cache ala SnapKV dengan budget per layer |
| `niah.py` | Pembuat data Needle-in-a-Haystack, recall ketat, dan skor parsial |
| `run.py` | Menjalankan pencarian dan evaluasi akhir untuk setiap seed |
| `plot.py` | Merangkum semua seed menjadi tabel dan grafik |
| `test_evolkv.py`, `test_kv_cache.py` | Tes unit |

## Perbedaan dengan paper

- Optimizer-nya Genetic Algorithm, bukan CMA-ES.
- Model Qwen2.5-0.5B (24 layer) dengan NIAH sintetis, bukan Mistral/Llama dengan LongBench atau RULER.
- Kandidat diskalakan agar rata-rata budget tepat `c` sebelum dievaluasi. Paper memakai CacheScore sebagai penalti
  dan menyesuaikan budget setelah pencarian. Akibatnya CacheScore di sini selalu 1 dan tidak lagi membedakan kandidat.
- Fitness memakai skor parsial per digit, sedangkan evaluasi akhir tetap memakai recall ketat.

## Batasan

- Hasil antar seed masih bervariasi cukup besar, misalnya 0,089–0,267 pada c=128, karena fitness hanya dihitung dari
  10 sampel latih. Paper memakai 30 sampel.
- Budget layer 0–1 berbeda jauh antar seed (32 sampai 349), jadi bentuk alokasi di layer bawah belum stabil. Pola yang
  muncul di ketiga seed hanya budget tinggi di layer 16 (386–426 token, tiga kali target).
- PyramidKV jauh di bawah SnapKV pada data ini. Penyebab yang paling mungkin, tetapi belum terbukti: layer bawah,
  yang mendapat budget terbesar di PyramidKV, jarang menangkap needle.
- Baseline StreamingLLM belum tersedia.
- `kv_cache.py` mengikuti API internal transformers 5.22.1, jadi versinya dikunci di `requirements.txt`.
  Hanya model dengan struktur `model.layers` seperti Qwen2 yang sudah diuji.
