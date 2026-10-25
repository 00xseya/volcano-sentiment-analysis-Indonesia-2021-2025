#!/usr/bin/env python3
"""
Konversi export Label Studio (JSON) menjadi dataset sentimen final.

Label Studio choices yang dipakai:
  "0 - Negatif" -> 0
  "1 - Netral"  -> 1
  "2 - Positif" -> 2

Contoh:
  .venv/bin/python script/sentiment/export_to_dataset.py \
      --export data/sentiment/label_studio/export.json \
      --output data/sentiment/labeled/sentiment_labels_final.csv
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def parse_choice(value: str) -> int | None:
    match = re.match(r"\s*([012])\b", str(value))
    if match:
        return int(match.group(1))
    lowered = str(value).strip().lower()
    if lowered.startswith("neg"):
        return 0
    if lowered.startswith("neu"):
        return 1
    if lowered.startswith("pos"):
        return 2
    return None


def extract_label(task: dict) -> tuple[int | None, str, str]:
    """Ambil label dari annotation manusia, fallback ke prediction."""
    annotations = task.get("annotations") or []
    for ann in annotations:
        if ann.get("was_cancelled"):
            continue
        for result in ann.get("result", []):
            choices = result.get("value", {}).get("choices") or []
            if choices:
                label = parse_choice(choices[0])
                if label is not None:
                    return label, "human", str(ann.get("completed_by", ""))

    for pred in task.get("predictions") or []:
        for result in pred.get("result", []):
            choices = result.get("value", {}).get("choices") or []
            if choices:
                label = parse_choice(choices[0])
                if label is not None:
                    return label, "prediction", str(pred.get("model_version", ""))

    return None, "unlabeled", ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", required=True, help="File JSON export Label Studio")
    parser.add_argument(
        "--output",
        default="data/sentiment/labeled/sentiment_labels_final.csv",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    export_path = Path(args.export)
    output_path = Path(args.output)
    if not export_path.exists():
        print(f"[error] Export tidak ditemukan: {export_path}")
        return 1

    with open(export_path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("tasks", [])

    records = []
    for task in data:
        label, source, annotator = extract_label(task)
        row = dict(task.get("data", {}))
        row.update({
            "label": label,
            "label_name": {0: "negatif", 1: "netral", 2: "positif"}.get(label, ""),
            "label_source": source,
            "annotator": annotator,
            "prediction_score": (task.get("predictions") or [{}])[0].get("score", ""),
            "meta_confidence": task.get("meta", {}).get("confidence", ""),
            "meta_method": task.get("meta", {}).get("method", ""),
        })
        records.append(row)

    out = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False, encoding="utf-8")

    labeled = out[out["label"].notna()]
    print(f"[output] {output_path} ({len(out):,} task, {len(labeled):,} berlabel)")
    if len(labeled):
        counts = labeled["label_name"].value_counts().reindex(
            ["negatif", "netral", "positif"], fill_value=0
        )
        print(counts.to_string())
        print("\nSumber label:")
        print(labeled["label_source"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
