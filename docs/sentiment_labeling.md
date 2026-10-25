# Panduan Labeling Sentimen (0 / 1 / 2)

Dokumen ini adalah *codebook* dan catatan metode untuk memberi label sentimen
pada satu kalimat komentar YouTube tentang erupsi gunung di Indonesia
(2021–2025). Label yang dipakai di dataset: **0 = negatif, 1 = netral,
2 = positif**.

## 1. Skema label

| Label | Nama | Arti |
|------:|------|------|
| `0` | **Negatif** | Kalimat mengekspresikan emosi/sikap negatif: kritik, keluhan, kemarahan, kekecewaan, ketakutan, kekhawatiran, kesedihan, duka, hinaan, atau penilaian buruk. |
| `1` | **Netral** | Kalimat informatif/faktual, pertanyaan, sapaan, permintaan, spam/off-topic, atau tidak mengandung polaritas yang jelas. |
| `2` | **Positif** | Kalimat mengekspresikan emosi/sikap positif: pujian, terima kasih, dukungan, harapan/doa baik, rasa syukur, kekaguman, humor ringan, atau penilaian baik. |

Label disimpan sebagai angka di CSV/JSONL. Di Label Studio, pilihan ditampilkan
sebagai `0 - Negatif`, `1 - Netral`, dan `2 - Positif` agar hotkey dan angka
tetap terlihat.

## 2. Aturan keputusan

1. **Fokus pada sikap penulis**, bukan pada topiknya.
   - `"gunung merapi meletus lagi"` → **netral** (pernyataan fakta).
   - `"erupsi ini mengerikan sekali"` → **negatif** (evaluasi/emosi).
2. **Kata topik tidak otomatis negatif.** Kata `bencana`, `erupsi`, `meletus`,
   `korban`, `lahar` hanya menandakan topik. Lihat kata evaluatif di sekitarnya.
3. **Kutipan berita / laporan faktual** → **netral**, walaupun isinya peristiwa
   buruk. Contoh: `"sebanyak 12 orang meninggal"` adalah laporan, bukan duka.
4. **Doa dan harapan baik** → **positif**.
   - `"semoga tidak ada korban jiwa"` → **positif** (harapan baik).
   - `"mudah-mudahan aman"` → **positif**.
5. **Duka dan belasungkawa** → **negatif**.
   - `"turut berduka cita"`, `"innalillahi"`, `"rest in peace"` → **negatif**.
   - Doa setelah pernyataan duka mengikuti emosi dominan kalimat.
6. **Kritik dan argumen** → **negatif**, termasuk kritik terhadap pemerintah,
   BNPB, SAR, kreator, atau komentar orang lain.
7. **Humor/ejekan ringan**:
   - `"wkwkwk"`, `"lucu"`, `"kocak"` tanpa sasaran hinaan → **positif**.
   - Ejekan yang merendahkan orang/suku/kelompok → **negatif**.
8. **Sarkasme** dinilai dari makna yang dimaksud, bukan kata permukaan.
9. **Kalimat campuran** (mis. pujian + kritik): pilih emosi yang paling dominan
   atau yang menjadi tujuan utama penulis.
10. **Tidak dapat dipahami / terlalu pendek / singkatan acak** → **netral**
    (mis. `"ha"`, `"iyq"`, `"sagh"`).
11. **Spam, promosi, judi online, dan off-topic** → **netral**.

## 3. Contoh berlabel

| Teks | Label | Alasan |
|------|------:|--------|
| `mantap sekali penjelasannya` | 2 | Pujian |
| `terima kasih dokumentasinya bang` | 2 | Terima kasih |
| `semoga tidak ada korban jiwa` | 2 | Doa/harapan baik |
| `alhamdulillah aman semua` | 2 | Rasa syukur |
| `pemerintah tidak peduli rakyat kecil` | 0 | Kritik |
| `serem banget erupsinya` | 0 | Ketakutan |
| `turut berduka cita untuk korban` | 0 | Duka |
| `tolol` | 0 | Hinaan |
| `gunung merapi meletus lagi` | 1 | Fakta |
| `apa itu awan panas?` | 1 | Pertanyaan |
| `kak bahas gunung toba dong` | 1 | Permintaan |
| `siapa yang nonton 2026` | 1 | Netral/off-topic |

## 4. Sumber data

- Utama: `data/cleansing/youtube_comments_cleansed.csv` (64.836 baris,
  14 kolom). Kolom teks: `text_cleansed`; metadata yang dipertahankan:
  `comment_id`, `author`, `video_title`, `video_year`, `votes`.
- Alternatif: `data/ner_distribution/data_text.csv` (64.836 baris, hanya
  `video_title`, `video_year`, `text_cleansed`). Isinya sama; versi cleansing
  dipakai karena metadata tambahan berguna saat review di Label Studio.

Tidak diperlukan output NER IndoBERT untuk labeling ini. Label sentimen
diberikan pada **seluruh kalimat**, bukan pada entity. Metadata video hanya
membantu reviewer memahami konteks, bukan fitur model.

## 5. Metode yang dipakai

### 5.1 `rules_v1` — leksikon dan aturan domain

Implementasi: `script/sentiment/sentiment_rules.py` dan
`script/sentiment/label_rules.py`.

Tahapan per baris:

1. Normalisasi teks (lowercase, rapikan spasi/tanda baca).
2. **Override presisi tinggi** (`strong_label`), berurutan:
   - teks pendek yang sudah jelas (`mantap`, `wkwkwk`, `tolol`, `ha`);
   - duka/belasungkawa (`innalillahi`, `turut berduka`, `alfatihah`);
   - kata kesedihan/takut (`sedih`, `takut`, `khawatir`);
   - negatif presisi tinggi (hinaan, kritik, `tidak peduli`, `hoax`);
   - doa/harapan baik (`semoga`, `mudah-mudahan`, `semoga sehat`);
   - positif presisi tinggi (pujian, terima kasih, humor);
   - pertanyaan/permintaan informasi (`apa`, `bagaimana`, `ingin tahu`).
3. Jika tidak ada override, hitung skor leksikon positif/negatif dengan
   pembobotan:
   - frasa kuat = bobot besar;
   - kata topik/bencana = bobot kecil atau nol;
   - negasi (`tidak`, `bukan`, `jangan`, `kurang`) membalik polaritas;
   - pengecualian seperti `tidak apa-apa`, `tidak ada korban`, `tidak sabar`.
4. Jika margin skor < 0,6, label jatuh ke **netral** (tidak ada sinyal cukup).
5. Simpan `confidence`, `method`, `rule_margin`, `rule_hits`, dan
   `needs_review`.

Nilai `method`:

| Method | Arti |
|--------|------|
| `rule-override` | Diputuskan oleh aturan presisi tinggi (confidence 0,95). |
| `rule` | Diputuskan oleh skor leksikon dengan margin ≥ 0,6. |
| `neutral-default` | Tidak ada sinyal kuat; default netral. |

`confidence` bukan probabilitas terkalibrasi. Nilainya hanya untuk
memprioritaskan review: 0,55 (default netral) sampai 0,95 (override).

### 5.2 Model transformer (eksperimen, bukan pipeline final)

`script/sentiment/label_sentiment.py` masih tersedia untuk eksperimen:

- `w11wo/indonesian-roberta-base-sentiment-classifier`;
- `Aardiiiiy/indobertweet-base-Indonesian-sentiment-analysis`.

Pada uji 120 kalimat, model terbaik hanya mencapai akurasi **51,7%**, sedangkan
gabungan model+aturan **58,3%**, dan aturan saja **84,2%**. Model juga bias
negatif pada teks pendek dan komentar bernada positif. Karena itu pipeline
final memakai aturan sebagai penentu, bukan model. Model dapat dijalankan lagi
untuk analisis error atau ensemble di masa depan, tetapi bukan sumber label
default.

## 6. Evaluasi kualitas

Dua sampel manual masing-masing 120 baris:

| Sampel | Akurasi aturan | Catatan |
|--------|---------------:|---------|
| `data/sentiment/gold/gold_sample.csv` | 84,2% | Dipakai untuk tuning awal. |
| `data/sentiment/gold/gold_sample_holdout.csv` | 85,0% | Holdout, tetapi sempat dipakai untuk menambah leksikon umum. |

Angka ini **optimistis** karena holdout tidak lagi sepenuhnya bersih setelah
iterasi aturan. Untuk klaim final, gunakan sampel baru yang belum pernah
dilihat. Kesalahan yang masih umum:

- sarkasme dan kritik tidak langsung;
- bahasa daerah/plesetan yang tidak ada di leksikon;
- kalimat netral yang mengandung satu kata positif/negatif;
- kalimat campuran (pujian + kritik);
- singkatan baru dan nama tokoh.

Karena itu review manusia tetap diperlukan, terutama untuk baris dengan
`confidence` rendah.

## 7. Alur Label Studio

Label Studio dipakai sebagai alat review pre-annotation, bukan untuk mengetik
64 ribu label dari nol.

### 7.0 Status proyek saat ini

- Server: `http://localhost:8080` (jalankan ulang dengan perintah di 7.1).
- Project: **Volcano Sentiment 2021–2025** (project ID 2).
- Task: 64.836, semuanya sudah punya prediction `rules_v1`.
- Akun lokal: `admin@example.com` (password dikirim terpisah; ganti bila perlu).
- Urutan task: confidence terendah lebih dulu, jadi review dimulai dari baris
  paling ragu.

### 7.1 Jalankan Label Studio

Instalasi terisolasi sudah disiapkan di `.venv-labelstudio`:

```bash
.venv-labelstudio/bin/label-studio start --port 8080 --no-browser
```

Buka `http://localhost:8080`. Jika belum ada akun, buat akun lokal.

### 7.2 Buat project

1. **Create Project** → nama: `Volcano Sentiment 2021–2025`.
2. **Labeling Setup** → **Code** → tempel isi
   `data/sentiment/label_studio/config.xml`.
3. Simpan.

### 7.3 Import task

Urutan yang disarankan:

1. `tasks_priority.json` — 4.197 baris paling ambigu (prediksi aturan dengan
   margin rendah + sampel netral). Mulai dari sini agar cepat menemukan error.
2. `tasks_part_01.json` … `tasks_part_07.json` — seluruh 64.836 baris, masing-
   masing 10.000 (part terakhir 4.836).
3. `tasks_all.json` — alternatif satu file besar (40 MB), jika ingin import
   sekaligus.

Setiap task sudah membawa **prediction** `rules_v1`. Prediction muncul sebagai
pre-annotation; reviewer tinggal:

- **Submit** jika label sudah benar;
- tekan `1` / `2` / `3` untuk mengganti label, lalu **Submit**.

Hotkey: `1` = Negatif, `2` = Netral, `3` = Positif.

### 7.4 Export dan konversi

1. Di project, klik **Export** → **JSON**.
2. Simpan ke `data/sentiment/label_studio/export.json`.
3. Konversi ke dataset final:

```bash
.venv/bin/python script/sentiment/export_to_dataset.py \
    --export data/sentiment/label_studio/export.json \
    --output data/sentiment/labeled/sentiment_labels_final.csv
```

Skrip memprioritaskan annotation manusia; jika task belum direview, prediction
`rules_v1` dipakai sebagai fallback. Kolom `label_source` menandai
`human`/`prediction`.

## 8. Output

| Path | Isi |
|------|-----|
| `data/sentiment/labeled/sentiment_labels_rules.csv` | 64.836 baris + label, confidence, method, rule hits. |
| `data/sentiment/labeled/sentiment_labels_rules.jsonl` | Format JSONL. |
| `data/sentiment/labeled/label_distribution.csv` | Distribusi total, per tahun, per method. |
| `data/sentiment/labeled/rules_metadata.json` | Metadata pipeline. |
| `data/sentiment/review/review_queue.csv` | 34.800 baris yang disarankan dicek. |
| `data/sentiment/label_studio/config.xml` | Konfigurasi UI Label Studio. |
| `data/sentiment/label_studio/tasks_*.json` | Task + pre-annotation. |
| `data/sentiment/gold/gold_sample.csv` | Sampel tuning 120 baris. |
| `data/sentiment/gold/gold_sample_holdout.csv` | Sampel holdout 120 baris. |
| `data/sentiment/gold/eval_rules.json` | Akurasi dan confusion matrix rules_v1. |

## 9. Reproduksi

```bash
# 1. Buat label aturan untuk seluruh data
.venv/bin/python script/sentiment/label_rules.py \
    --input data/cleansing/youtube_comments_cleansed.csv \
    --output-dir data/sentiment/labeled

# 2. Buat task Label Studio + pre-annotation
.venv/bin/python script/sentiment/export_label_studio.py \
    --input data/sentiment/labeled/sentiment_labels_rules.csv \
    --output-dir data/sentiment/label_studio

# 3. Setelah review di Label Studio, konversi export
.venv/bin/python script/sentiment/export_to_dataset.py \
    --export data/sentiment/label_studio/export.json \
    --output data/sentiment/labeled/sentiment_labels_final.csv

# 4. Evaluasi aturan pada sampel gold
.venv/bin/python script/sentiment/evaluate_rules.py \
    --gold data/sentiment/gold/gold_sample.csv \
           data/sentiment/gold/gold_sample_holdout.csv \
    --output data/sentiment/gold/eval_rules.json
```

Pipeline model lama (opsional):

```bash
.venv/bin/python script/sentiment/label_sentiment.py \
    --input data/ner_distribution/data_text.csv \
    --output-dir /tmp/sentiment_model
```

## 10. Catatan etika dan keterbatasan

- Label mencerminkan sikap penulis komentar, bukan kebenaran faktual tentang
  bencana.
- Komentar YouTube rentan terhadap spam, ajakan judi, dan ujaran kebencian.
  Label `0` untuk hinaan bukan berarti konten tersebut layak disebarkan.
- Jangan menganggap `confidence` sebagai probabilitas. Gunakan hanya untuk
  triage.
- Dataset gold dibuat dengan bantuan AI dan review manual terbatas; sebutkan
  hal ini di laporan agar tidak menyesatkan.
