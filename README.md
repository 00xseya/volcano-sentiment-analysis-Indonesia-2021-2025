# Analisis Sentimen Publik terhadap Erupsi Gunung Api di Indonesia (2021–2025)

**Mata Kuliah:** Pengolahan Bahasa Alami (A)  
**Dosen Pengampu:** Assoc. Prof. Dr. Ir. Aris Tjahyanto, M.Kom  
**Tanggal Pengumpulan:** 5 November 2026

| Nama | NRP | Peran |
|------|-----|-------|
| Aldani | — | Data Preparation (scraping, cleaning, cleansing) |
| Muhammad Sahril | — | First Analysis — anotasi NER, fine-tuning IndoBERT |
| Pankop / Naufal | — | Second Analysis — rule-based sentiment labeling |
| Azhar Aditya Pratama | 5026231084 | Evaluation — NER test eval, notebook presentasi |

---

## Tautan Pengumpulan & Berkas Penting

| Komponen | Tautan / Berkas |
|---|---|
| **Repositori GitHub** | [github.com/00xseya/volcano-sentiment-analysis-Indonesia-2021-2025](https://github.com/00xseya/volcano-sentiment-analysis-Indonesia-2021-2025) |
| **Video Presentasi (YouTube Unlisted)** | *[Tempel link YouTube Unlisted di sini]* |
| **Slide Presentasi (PPTX)** | [Presentasi-Volcano-Sentiment-Analysis-Final.pptx](./Presentasi-Volcano-Sentiment-Analysis-Final.pptx) |
| **Notebook Demo & Analisis** | [notebook/final_analysis.ipynb](./notebook/final_analysis.ipynb) |
| **Dataset Berlabel (64.836 baris)** | [data/sentiment/labeled/sentiment_labels_rules.csv](./data/sentiment/labeled/sentiment_labels_rules.csv) |
| **Model Weights & Backup Data** | [Google Drive Folder](https://drive.google.com/drive/folders/1rPPRFv0xOuI1d_6hZtACc1iHRCy-wSp2?usp=drive_link) |

---

## Deskripsi Proyek

Penelitian NLP untuk menganalisis sentimen publik Indonesia terhadap erupsi gunung api (2021–2025) dari komentar YouTube. Pendekatan dua tahap:

1. **NER** — Fine-tuning IndoBERT untuk ekstraksi entitas vulkanik (VOL, LOC, HAZ, TIME, GOV, SOC)
2. **Analisis Sentimen** — Pipeline berbasis aturan untuk pelabelan negatif / netral / positif

**Repositori:** https://github.com/00xseya/volcano-sentiment-analysis-Indonesia-2021-2025

---

## Panduan Membaca Proyek (untuk Dosen)

### 1. Mulai dari Presentasi
Buka file presentasi untuk gambaran lengkap:  
👉 **[Presentasi-Volcano-Sentiment-Analysis-Final.pptx](./Presentasi-Volcano-Sentiment-Analysis-Final.pptx)**


### 2. Lihat Notebook Utama
Notebook `notebook/final_analysis.ipynb` berisi seluruh analisis yang sudah dijalankan — semua cell sudah ter-execute dengan output lengkap:
- Statistik dataset
- Distribusi entitas NER
- Hasil evaluasi model (validation & test set)
- Distribusi sentimen per tahun
- Knowledge Graph co-occurrence entitas

```bash
# Jalankan Jupyter untuk membuka notebook interaktif:
pip install -r requirements.txt
jupyter lab notebook/final_analysis.ipynb
```

### 3. Pipeline Lengkap

```
Scraping (yt-dlp)
  → 335 video, 87.840 komentar raw          [data/raw/]
  → Cleaning: 80.239 baris                  [data/cleaning/]
  → Cleansing: 79.524 baris                 [data/cleansing/]
  → Corpus NER: 64.836 baris                [data/ner_distribution/data_text.csv]
  → Anotasi manual Label Studio: 18.537     [data/ner_distribution/annotated/]
  → Split: 55.101 train / 3.054 val / 6.681 test
  → Fine-tuning IndoBERT (5 epoch)          [models/ — lihat catatan di bawah]
  → Evaluasi NER test set: Micro F1 97,83%  [evaluation/]
  → Labeling sentimen: 64.836 komentar      [data/sentiment/labeled/]
  → Notebook presentasi terintegrasi        [notebook/final_analysis.ipynb]
```

---

## Struktur Direktori

```
├── notebook/
│   ├── final_analysis.ipynb      ← NOTEBOOK UTAMA (jalankan ini)
│   ├── preprocessing.ipynb       ← Preprocessing awal
│   ├── data_distribution.ipynb   ← BIO alignment & split
│   └── modelling.ipynb           ← Fine-tuning IndoBERT
│
├── data/
│   ├── raw/                      ← 87.840 komentar mentah
│   ├── cleaning/                 ← 80.239 setelah cleaning
│   ├── cleansing/                ← 79.524 setelah cleansing
│   ├── ner_distribution/
│   │   ├── data_text.csv         ← 64.836 corpus NER
│   │   ├── annotated/            ← Hasil anotasi Label Studio
│   │   └── data_distribution/   ← train/val/test.jsonl
│   └── sentiment/
│       └── labeled/
│           └── sentiment_labels_rules.csv  ← DATASET BERLABEL (64.836 baris)
│
├── evaluation/
│   ├── run_ner_eval.py           ← Script evaluasi NER pada test set
│   └── ner_eval_results.json     ← Hasil: Micro F1 97,83%
│
├── docs/
│   ├── sentiment_labeling.md     ← Codebook aturan pelabelan sentimen
│   └── images/                  ← Diagram arsitektur proyek
│
├── Presentasi-Volcano-Sentiment-Analysis-Final.pptx  ← File presentasi
└── requirements.txt
```

---

## Dataset Berlabel

File utama: `data/sentiment/labeled/sentiment_labels_rules.csv`

- **64.836 baris** komentar YouTube berlabel sentimen
- Kolom: `comment_id`, `author`, `video_title`, `video_year`, `text`, `label`, `label_name`, `confidence`, `method`, `rule_hits`, `needs_review`
- Label: `0 = negatif`, `1 = netral`, `2 = positif`

| Label | Jumlah | % |
|-------|--------|---|
| Negatif (0) | 7.059 | 10,9% |
| Netral (1) | 37.567 | 57,9% |
| Positif (2) | 20.210 | 31,2% |

---

## Hasil Evaluasi

### NER — IndoBERT Fine-tuned

| Entitas | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| VOL | 0,9798 | 0,9887 | **0,9842** | 884 |
| SOC | 0,9966 | 0,9916 | **0,9941** | 594 |
| LOC | 0,9600 | 0,9746 | **0,9673** | 394 |
| GOV | 0,9648 | 0,9384 | **0,9514** | 146 |
| TIME | 0,9421 | 0,9580 | **0,9500** | 119 |
| HAZ | 0,8873 | 0,9844 | **0,9333** | 64 |
| **Micro Avg** | **0,9747** | **0,9818** | **0,9783** | **2.201** |

### Sentimen — Rule-based

| Set Evaluasi | Ukuran | Akurasi | Macro F1 |
|-------------|--------|---------|----------|
| Dev set | 120 baris | **84,17%** | 0,842 |
| Holdout set | 120 baris | **85,00%** | 0,837 |

---

## Catatan: Model Weights & File Besar

Bobot model IndoBERT (`models/`) dan beberapa file data besar tidak di-commit ke Git karena ukurannya. Tersedia di Google Drive:

- **Annotated data & model output:** [Google Drive](https://drive.google.com/drive/folders/1rPPRFv0xOuI1d_6hZtACc1iHRCy-wSp2?usp=drive_link)
- **Diagram arsitektur:** [Google Drive](https://drive.google.com/file/d/1gxrJOELYaf9mzmR1Bn9X4fHdlL7vATg2/view?usp=sharing)

Untuk menjalankan `evaluation/run_ner_eval.py`, unduh folder `models/` dari Drive dan letakkan di root repo.

---

## Cara Menjalankan

```bash
# Setup environment (Python 3.11)
python -m venv .venv
source .venv/bin/activate
pip install torch transformers seqeval pandas numpy matplotlib seaborn wordcloud networkx jupyter

# Jalankan notebook utama
jupyter lab notebook/final_analysis.ipynb

# Re-run evaluasi NER (butuh folder models/)
python evaluation/run_ner_eval.py
```
