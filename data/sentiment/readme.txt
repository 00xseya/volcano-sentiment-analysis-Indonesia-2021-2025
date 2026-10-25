SENTIMENT LABELING
==================

Folder ini berisi hasil labeling sentimen per kalimat dari 64.836 komentar
YouTube tentang erupsi gunung di Indonesia (2021-2025).

Sumber data:
  data/cleansing/youtube_comments_cleansed.csv
  (kolom teks: text_cleansed; metadata: comment_id, author, video_title,
  video_year, votes)

Skema label:
  0 = negatif
  1 = netral
  2 = positif

Isi folder:
  labeled/sentiment_labels_rules.csv   Dataset lengkap (64.836 baris) + label,
                                      confidence, method, rule_hits,
                                      needs_review.
  labeled/sentiment_labels_rules.jsonl Format JSONL dari dataset yang sama.
  labeled/label_distribution.csv      Distribusi label total, per tahun, dan
                                      per metode.
  labeled/rules_metadata.json         Metadata pipeline rules_v1.
  review/review_queue.csv             Baris yang disarankan dicek manual
                                      (34.800 baris; prioritas confidence
                                      rendah).
  label_studio/config.xml              Konfigurasi UI Label Studio (3 pilihan).
  label_studio/tasks_priority.json     4.197 task prioritas review.
  label_studio/tasks_part_01.json ...   Seluruh task, dipecah 10.000 per file.
  label_studio/tasks_all.json          Semua task dalam satu file (40 MB).
  label_studio/README.md               Panduan singkat Label Studio.
  gold/gold_sample.csv                 Sampel tuning 120 baris.
  gold/gold_sample_holdout.csv         Sampel holdout 120 baris.
  gold/gold_label_map.json             Definisi label gold.
  gold/eval_rules.json                 Akurasi + confusion matrix rules_v1.

Metode:
  rules_v1 = leksikon + aturan domain (doa, duka, humor, hinaan, negasi,
  pertanyaan, kata topik bencana). Model transformer hanya untuk eksperimen;
  akurasinya lebih rendah daripada aturan pada uji 2x120 sampel.

Cara membuat ulang:
  .venv/bin/python script/sentiment/label_rules.py \
      --input data/cleansing/youtube_comments_cleansed.csv \
      --output-dir data/sentiment/labeled

  .venv/bin/python script/sentiment/export_label_studio.py \
      --input data/sentiment/labeled/sentiment_labels_rules.csv \
      --output-dir data/sentiment/label_studio

Cara review di Label Studio:
  .venv-labelstudio/bin/label-studio start --port 8080 --no-browser

  Import config.xml sebagai labeling setup, lalu import tasks_priority.json
  atau tasks_part_*.json. Setelah selesai, export JSON dan jalankan:
  .venv/bin/python script/sentiment/export_to_dataset.py \
      --export data/sentiment/label_studio/export.json \
      --output data/sentiment/labeled/sentiment_labels_final.csv

Panduan label lengkap: docs/sentiment_labeling.md
