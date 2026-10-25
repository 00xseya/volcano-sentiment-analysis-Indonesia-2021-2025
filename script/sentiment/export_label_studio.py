#!/usr/bin/env python3
"""
Buat proyek Label Studio untuk labeling sentimen 3 kelas.

Output:
  data/sentiment/label_studio/config.xml
  data/sentiment/label_studio/tasks_all.json
  data/sentiment/label_studio/tasks_part_XX.json (pecahan 10k)
  data/sentiment/label_studio/tasks_priority.json (confidence terendah)
  data/sentiment/label_studio/README.md

Import file JSON ke Label Studio, lalu review pre-annotation. Hotkey:
1 = Negatif, 2 = Netral, 3 = Positif.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

CHOICE_VALUES = {
    0: "0 - Negatif",
    1: "1 - Netral",
    2: "2 - Positif",
}

CONFIG_XML = """<View>
  <Header value="Label sentimen kalimat (0=Negatif, 1=Netral, 2=Positif)"/>
  <Text name="text" value="$text" granularity="word"/>
  <Choices name="sentiment" toName="text" choice="single" showInline="true">
    <Choice value="0 - Negatif" hotkey="1"/>
    <Choice value="1 - Netral" hotkey="2"/>
    <Choice value="2 - Positif" hotkey="3"/>
  </Choices>
  <View style="margin-top: 8px; color: #666; font-size: 12px;">
    <Text name="meta" value="$meta_text"/>
  </View>
</View>
"""

README = """# Label Studio — Sentiment Labeling

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
.venv/bin/python script/sentiment/export_to_dataset.py \
    --export data/sentiment/label_studio/export.json \
    --output data/sentiment/labeled/sentiment_labels_final.csv
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
"""


def task_from_row(row: pd.Series) -> dict:
    label = int(row["label"])
    return {
        "data": {
            "row_id": int(row["row_id"]),
            "comment_id": str(row.get("comment_id", "")),
            "author": str(row.get("author", "")),
            "video_title": str(row.get("video_title", "")),
            "video_year": str(row.get("video_year", "")),
            "text": str(row["text"]),
            "confidence": float(row.get("confidence", 0)),
            "method": str(row.get("method", "")),
            "needs_review": bool(row.get("needs_review", False)),
            "meta_text": (
                f"{row.get('video_year', '')} | {row.get('author', '')} | "
                f"confidence={row.get('confidence', '')} | {row.get('method', '')}"
            ),
        },
        "meta": {
            "confidence": float(row.get("confidence", 0)),
            "method": str(row.get("method", "")),
            "needs_review": bool(row.get("needs_review", False)),
        },
        "predictions": [{
            "model_version": "rules_v1",
            "score": float(row.get("confidence", 0)),
            "result": [{
                "from_name": "sentiment",
                "to_name": "text",
                "type": "choices",
                "value": {"choices": [CHOICE_VALUES[label]]},
            }],
        }],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="data/sentiment/labeled/sentiment_labels_rules.csv",
    )
    parser.add_argument("--output-dir", default="data/sentiment/label_studio")
    parser.add_argument("--chunk-size", type=int, default=10000)
    parser.add_argument("--priority-size", type=int, default=3000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    out_dir = Path(args.output_dir)
    if not input_path.exists():
        print(f"[error] Input tidak ditemukan: {input_path}")
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    print(f"[input] {len(df):,} baris")
    df = df.sort_values("confidence", ascending=True).reset_index(drop=True)

    # Prioritas review: prediksi aturan yang belum yakin + sampel neutral-default.
    ambiguous = df[(df["method"] == "rule") & (df["confidence"] < 0.85)]
    neutral_sample = df[df["method"] == "neutral-default"].sample(
        n=min(2000, int((df["method"] == "neutral-default").sum())),
        random_state=42,
    )
    priority_df = (
        pd.concat([ambiguous, neutral_sample])
        .drop_duplicates(subset=["row_id"])
        .sort_values("confidence")
    )
    priority_rows = priority_df.to_dict(orient="records")

    tasks = [task_from_row(row) for _, row in df.iterrows()]

    (out_dir / "config.xml").write_text(CONFIG_XML, encoding="utf-8")
    (out_dir / "README.md").write_text(README, encoding="utf-8")

    all_path = out_dir / "tasks_all.json"
    with open(all_path, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False)

    for i in range(0, len(tasks), args.chunk_size):
        part = tasks[i:i + args.chunk_size]
        path = out_dir / f"tasks_part_{i // args.chunk_size + 1:02d}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(part, f, ensure_ascii=False)
        print(f"[output] {path} ({len(part):,} task)")

    priority = [task_from_row(pd.Series(row)) for row in priority_rows]
    priority_path = out_dir / "tasks_priority.json"
    with open(priority_path, "w", encoding="utf-8") as f:
        json.dump(priority, f, ensure_ascii=False)
    print(f"[output] {priority_path} ({len(priority):,} task)")
    print(f"[output] {all_path} ({len(tasks):,} task)")
    print(f"[output] {out_dir / 'config.xml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
