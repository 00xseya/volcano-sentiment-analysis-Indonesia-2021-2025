#!/usr/bin/env python3
"""
Label sentimen seluruh komentar dengan aturan presisi tinggi (tanpa model).

Cocok untuk membuat pre-annotation Label Studio dengan cepat. Setiap baris
menyimpan label, confidence, metode, dan bukti frasa yang cocok.

Contoh:
  .venv/bin/python script/sentiment/label_rules.py \
      --input data/cleansing/youtube_comments_cleansed.csv \
      --output-dir data/sentiment/labeled

Output:
  sentiment_labels_rules.csv
  sentiment_labels_rules.jsonl
  label_distribution.csv
  review_queue.csv
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sentiment_rules import (  # noqa: E402
    ID_TO_NAME_ID,
    normalize_text,
    score_rules,
    strong_label,
)


def confidence_from_margin(margin: float, strong: bool) -> float:
    if strong:
        return 0.95
    if margin < 0.6:
        return 0.55
    return round(min(0.90, 0.55 + margin / 5.0), 3)


def label_row(text: str) -> dict:
    text = normalize_text(text)
    strong_id, reason = strong_label(text)
    result = score_rules(text)
    scores = result.as_vector()
    rule_id = max(range(3), key=lambda i: scores[i])
    margin = sorted(scores, reverse=True)[0] - sorted(scores, reverse=True)[1]

    if strong_id is not None:
        label = strong_id
        method = "rule-override"
        reason_text = f"strong:{reason}"
    elif margin < 0.6:
        label = 1
        method = "neutral-default"
        reason_text = "no-strong-signal"
    else:
        label = rule_id
        method = "rule"
        reason_text = ""

    hits = list(result.hits or [])
    if reason_text:
        hits.insert(0, reason_text)

    return {
        "label": label,
        "label_name": ID_TO_NAME_ID[label],
        "confidence": confidence_from_margin(margin, strong_id is not None),
        "method": method,
        "rule_margin": round(margin, 4),
        "rule_hits": "; ".join(hits),
        "needs_review": (
            strong_id is None and (margin < 0.8 or label != rule_id)
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="data/cleansing/youtube_comments_cleansed.csv",
        help="CSV input (default: data/cleansing/youtube_comments_cleansed.csv)",
    )
    parser.add_argument("--output-dir", default="data/sentiment/labeled")
    parser.add_argument("--text-column", default="text_cleansed")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    if not input_path.exists():
        print(f"[error] Input tidak ditemukan: {input_path}")
        return 1
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    if args.text_column not in df.columns:
        print(f"[error] Kolom {args.text_column!r} tidak ada. Kolom: {list(df.columns)}")
        return 1
    if args.limit:
        df = df.head(args.limit).copy()

    print(f"[input] {len(df):,} baris dari {input_path}")
    records = []
    for idx, row in df.iterrows():
        rec = label_row(row[args.text_column])
        record = {
            "row_id": int(idx),
            "comment_id": row.get("comment_id", ""),
            "author": row.get("author", ""),
            "video_title": row.get("video_title", ""),
            "video_year": row.get("video_year", ""),
            "votes": row.get("votes", ""),
            "text": str(row[args.text_column]),
            **rec,
        }
        records.append(record)

    out = pd.DataFrame(records)
    out_path = output_dir / "sentiment_labels_rules.csv"
    out.to_csv(out_path, index=False, encoding="utf-8")

    jsonl_path = output_dir / "sentiment_labels_rules.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    review = out[out["needs_review"]].sort_values("confidence")
    review_path = output_dir.parent / "review" / "review_queue.csv"
    review_path.parent.mkdir(parents=True, exist_ok=True)
    review.to_csv(review_path, index=False, encoding="utf-8")

    counts = out["label_name"].value_counts().reindex(
        ["negatif", "netral", "positif"], fill_value=0
    )
    dist = pd.DataFrame({
        "label_name": counts.index,
        "count": counts.values,
        "percentage": np.round(counts.values / len(out) * 100, 2),
    })
    with open(output_dir / "label_distribution.csv", "w", encoding="utf-8") as f:
        f.write("# Distribusi keseluruhan\n")
        dist.to_csv(f, index=False)
        f.write("\n# Distribusi per tahun\n")
        out.groupby(["video_year", "label_name"]).size().unstack(fill_value=0).to_csv(f)
        f.write("\n# Metode\n")
        out.groupby(["method", "label_name"]).size().unstack(fill_value=0).to_csv(f)

    meta = {
        "pipeline": "rules_v1",
        "input": str(input_path),
        "rows": int(len(out)),
        "review_rows": int(len(review)),
        "label_counts": {k: int(v) for k, v in counts.items()},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(output_dir / "rules_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[output] {out_path}")
    print(f"[output] {jsonl_path}")
    print(f"[output] {review_path} ({len(review):,} review)")
    print(dist.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
