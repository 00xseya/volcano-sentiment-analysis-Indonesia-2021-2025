# Label Studio — Sentiment Labeling

## Cara pakai (cepat)

1. Jalankan Label Studio:
   ```bash
   .venv-labelstudio/bin/label-studio start --port 8080 --no-browser
   ```
   Buka `http://localhost:8080`. Akun lokal yang sudah disiapkan:
   `admin@example.com` (password dikirim terpisah; ganti setelah dipakai).

2. Buka project:
   - Project **Volcano Sentiment 2021–2025** (ID 2) sudah dibuat dan seluruh
     64.836 task sudah diimpor.
   - Jika membuat project baru: **Labeling Setup** → **Code** → tempel isi
     `config.xml`, lalu import `tasks_part_01.json` … `tasks_part_07.json`.

3. Review data:
   - Prediction otomatis muncul sebagai pre-annotation. Klik **Submit** jika
     sudah benar, atau ubah label dengan hotkey `1` / `2` / `3`.
   - Urutan task sudah dari confidence terendah, jadi mulai dari atas.

4. Export hasil:
   - **Export** → pilih **JSON**.
   - Simpan sebagai `data/sentiment/label_studio/export.json`.

## Konversi export ke dataset latih

```bash
.venv/bin/python script/sentiment/export_to_dataset.py     --export data/sentiment/label_studio/export.json     --output data/sentiment/labeled/sentiment_labels_final.csv
```

## Skema label

| Hotkey | Label | Arti |
|-------:|-------|------|
| 1 | `0 - Negatif` | kritik, keluhan, takut, sedih, duka, hinaan |
| 2 | `1 - Netral` | fakta, pertanyaan, sapaan, off-topic, tanpa polaritas |
| 3 | `2 - Positif` | pujian, terima kasih, doa/harapan, dukungan, humor |

Panduan lengkap: `docs/sentiment_labeling.md`.

## Catatan

- Pre-annotation dibuat dengan aturan leksikon domain (`rules_v1`), bukan
  model transformer. Akurasi uji pada 2×120 sampel manual sekitar 85%, jadi
  tetap perlu review manusia terutama pada `confidence` rendah.
- `tasks_priority.json` berisi prediksi aturan yang masih ambigu (confidence
  < 0,85) ditambah sampel `neutral-default`; ini yang paling perlu dicek.
- File pecahan dibuat agar import tidak terlalu berat. Import satu per satu.
