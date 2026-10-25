#!/usr/bin/env python3
"""Evaluasi aturan rules_v1 pada sampel gold.

Contoh:
  .venv/bin/python script/sentiment/evaluate_rules.py \
      --gold data/sentiment/gold/gold_sample.csv \
             data/sentiment/gold/gold_sample_holdout.csv \
      --output data/sentiment/gold/eval_rules.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sentiment_rules import rule_label  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", nargs="+", required=True)
    parser.add_argument("--output", default="data/sentiment/gold/eval_rules.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = {}
    for path_str in args.gold:
        path = Path(path_str)
        df = pd.read_csv(path)
        if not {"gold_label", "text"}.issubset(df.columns):
            raise SystemExit(f"{path} harus punya kolom gold_label dan text")
        y_true = df["gold_label"].astype(int).to_numpy()
        y_pred = [rule_label(text)[0] for text in df["text"]]
        names = ["negatif", "netral", "positif"]
        report[path.name] = {
            "rows": int(len(df)),
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "confusion_matrix": confusion_matrix(
                y_true, y_pred, labels=[0, 1, 2]
            ).tolist(),
            "labels": names,
            "classification_report": classification_report(
                y_true, y_pred, labels=[0, 1, 2],
                target_names=names, output_dict=True, zero_division=0,
            ),
        }
        print(f"{path.name}: accuracy={report[path.name]['accuracy']:.4f} "
              f"({len(df)} baris)")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[output] {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
