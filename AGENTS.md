# Volcano Sentiment Analysis — Project Intelligence

> **Repo:** `/Users/azhar/Developer/volcano-sentiment-analysis-Indonesia-2021-2025`  
> **Last updated:** 2026-11-05  
> **Python target:** 3.11.18

---

## What This Project Is

NLP research on **Indonesian public sentiment toward volcanic eruptions (2021–2025)**, sourced from YouTube comments under eruption-related videos. The analysis is **NER-first**: named entities are extracted with a fine-tuned IndoBERT model (token classification), then sentiment is scored per comment. The scope covers major Indonesian volcanoes — Semeru, Merapi, Marapi, Krakatau, Lewotobi, Ruang, Ibu, Sinabung, and ~30 others.

Multi-contributor project:
- **Aldani** — Data preparation (dp)
- **Sahril** — First analysis / NER fine-tuning (fa)
- **Pankop / Naufal** — Second analysis / sentiment labeling (sa)
- **Azhar** — Evaluation & unified presentation (ev)

---

## Branches & Merges

| Branch | Status | Description |
|--------|--------|-------------|
| `main` | active | Fully merged with all data prep, NER, sentiment pipeline, and evaluation |
| `feature/data-preparation-youtube-comments` | merged | Original data pipeline commits |
| `feature/sentiment-labelling` | **merged** | Rule-based sentiment pipeline, gold eval, review queue |

---

## Pipeline Architecture

```
Source Collection      yt-dlp scrape → 335 videos, 5 years
        ↓
Raw Comments           87,840 rows  [data/raw/]
        ↓
Cleaning               Emoji removal, dedup, lowercase, missing values
                       80,239 rows  [data/cleaning/]
        ↓
Cleansing              Judol spam filter (50 removed), slang normalization,
                       typo fix, religious/negation harmonization
                       79,524 rows  [data/cleansing/]
        ↓
Column Extraction      video_title + video_year + text_cleansed
                       64,836 rows  [data/ner_distribution/data_text.csv]
        ↓
Pre-Annotation         Rule-based PHRASES dict → Label Studio JSON
                       [data/ner_distribution/pre_annotated_data.json]
        ↓
Human Annotation       Label Studio, 18,537/64,836 manually annotated (~28.6%)
                       [data/ner_distribution/annotated/annotated.json]
        ↓
BIO Token Alignment    IndoBERT tokenizer, char offsets → token labels
                       [notebook/data_distribution.ipynb]
        ↓
Train/Val/Test Split   55,101 / 3,054 / 6,681 tokenized records
                       [data/ner_distribution/data_distribution/]
        ↓
IndoBERT Fine-Tuning   indobert-base-p1, 13 NER labels, 5 epochs
                       [notebook/modelling.ipynb]
        ↓
NER Evaluation         Evaluated on test set (6,681 records) in 92.2s (MPS)
                       Micro F1: 97.83%
                       [evaluation/run_ner_eval.py → evaluation/ner_eval_results.json]
        ↓
Sentiment Labeling     Rule-based pipeline (rules_v1), 64,836 comments labeled
                       Accuracy 84.2% (dev) / 85.0% (holdout)
                       [data/sentiment/labeled/sentiment_labels_rules.csv]
        ↓
Unified Presentation   Complete code-first presentation notebook with live charts
                       [notebook/final_analysis.ipynb]
```

---

## Entity Schema (13 BIO Labels)

| Code | Full Name | Examples |
|------|-----------|---------|
| `VOL` | Volcano | gunung semeru, merapi, krakatau |
| `LOC` | Location | lumajang, yogyakarta, halmahera |
| `HAZ` | Hazard | awan panas, lahar hujan, debu vulkanik |
| `TIME` | Time | 7 desember 2021, dini hari |
| `GOV` | Government | PVMBG, BNPB, Basarnas, TNI |
| `SOC` | Social Actors | pendaki, pengungsi, relawan, netizen |

---

## NER Model Evaluation (Test Set Results)

Evaluated on `data/ner_distribution/data_distribution/test.jsonl` (6,681 records, 2,201 entity mentions) using `evaluation/run_ner_eval.py`:

| Entitas | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| **VOL** | 0.9798 | 0.9887 | **0.9842** | 884 |
| **SOC** | 0.9966 | 0.9916 | **0.9941** | 594 |
| **LOC** | 0.9600 | 0.9746 | **0.9673** | 394 |
| **GOV** | 0.9648 | 0.9384 | **0.9514** | 146 |
| **TIME** | 0.9421 | 0.9580 | **0.9500** | 119 |
| **HAZ** | 0.8873 | 0.9844 | **0.9333** | 64 |
| **Micro Avg** | **0.9747** | **0.9818** | **0.9783** | **2,201** |
| **Macro Avg** | **0.9551** | **0.9726** | **0.9634** | **2,201** |

---

## Sentiment Analysis (rules_v1 Results)

| Label | Meaning | Count | % |
|-------|---------|-------|---|
| `negatif` (0) | Kritik, duka, ketakutan, hinaan, kekecewaan | 7,059 | 10.9% |
| `netral` (1) | Fakta, pertanyaan, permintaan, spam/off-topic | 37,567 | 57.9% |
| `positif` (2) | Pujian, doa, terima kasih, rasa syukur, humor ringan | 20,210 | 31.2% |

**Sentiment by Year:**

| Year | Negatif | Netral | Positif |
|------|---------|--------|---------|
| 2021 | 874 | 4,535 | 3,859 |
| 2022 | 727 | 6,945 | 2,626 |
| 2023 | 3,035 | 10,464 | 7,068 |
| 2024 | 1,488 | 9,131 | 4,133 |
| 2025 | 935 | 6,492 | 2,524 |

**Accuracy on Gold Evaluation:**
- Dev Set (120 rows): **84.2%** accuracy (Macro F1: 0.842)
- Holdout Set (120 rows): **85.0%** accuracy (Macro F1: 0.837)

---

## Project Status

| Phase | Status | Deliverables / Files |
|-------|--------|----------------------|
| Data collection | ✅ Done | 335 videos, `data/source/` |
| Data extraction | ✅ Done | 87,840 raw comments, `data/raw/` |
| Cleaning | ✅ Done | 80,239 rows, `data/cleaning/` |
| Cleansing | ✅ Done | 79,524 rows (50 judol removed), `data/cleansing/` |
| Pre-annotation | ✅ Done | 64,836 tasks, `pre_annotated_data.json` |
| Human annotation | ✅ Done | 18,537 annotated, `annotated.json` |
| Split & BIO alignment | ✅ Done | 55,101 train / 3,054 val / 6,681 test |
| IndoBERT fine-tuning | ✅ Done | Weights in `models/` (gitignored, Drive backup) |
| NER Test Set Evaluation | ✅ Done | `evaluation/run_ner_eval.py`, `evaluation/ner_eval_results.json` |
| Sentiment labeling | ✅ Done | `data/sentiment/labeled/sentiment_labels_rules.csv` |
| Unified Presentation Notebook | ✅ Done | `notebook/final_analysis.ipynb` (executable, verified) |
| Documentation & Log | ✅ Done | `docs/ev-azhar.txt`, `docs/sentiment_labeling.md`, `AGENTS.md` |

---

## How to Run Presentation / Slides

1. **Jupyter Notebook (Interactive):**
   ```bash
   .venv/bin/jupyter lab notebook/final_analysis.ipynb
   ```
2. **Export to Reveal.js HTML Slides:**
   ```bash
   .venv/bin/jupyter nbconvert --to slides notebook/final_analysis.ipynb --post serve
   ```
3. **Re-run NER Evaluation:**
   ```bash
   .venv/bin/python evaluation/run_ner_eval.py
   ```

---

## Graph / Neo4j Exploration

- **Concept:** Knowledge Graph linking extracted entities (`VOL` - `LOC` - `HAZ` - `GOV` - `SOC`) with comment sentiment polarities.
- **Verdict on Neo4j:** Dedicated Neo4j DBMS server is overkill for static presentation.
- **Recommended Alternative:** In-notebook `networkx` + `pyvis` (or `matplotlib`) entity co-occurrence graph. Zero database setup, interactive HTML output directly inside Jupyter.

