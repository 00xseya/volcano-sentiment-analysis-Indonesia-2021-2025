"""
NER Evaluation — indobert-volcano-ner on test.jsonl
=====================================================
Runs the saved IndoBERT NER model on the held-out test split and writes
results to evaluation/ner_eval_results.json.

Usage:
    .venv/bin/python evaluation/run_ner_eval.py

Output:
    evaluation/ner_eval_results.json   — per-entity precision/recall/F1 + overall
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from transformers import AutoModelForTokenClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" if (ROOT / "models").exists() else ROOT / "indobert-volcano-ner"
TEST_FILE = ROOT / "data" / "ner_distribution" / "data_distribution" / "test.jsonl"
OUT_FILE = ROOT / "evaluation" / "ner_eval_results.json"

ID2LABEL = {
    0: "O", 1: "B-VOL", 2: "I-VOL", 3: "B-LOC", 4: "I-LOC",
    5: "B-HAZ", 6: "I-HAZ", 7: "B-TIME", 8: "I-TIME",
    9: "B-GOV", 10: "I-GOV", 11: "B-SOC", 12: "I-SOC",
}


def load_test(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        print("Device: MPS (Apple Silicon)")
        return torch.device("mps")
    if torch.cuda.is_available():
        print("Device: CUDA")
        return torch.device("cuda")
    print("Device: CPU")
    return torch.device("cpu")


def evaluate(batch_size: int = 64) -> dict:
    device = pick_device()

    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForTokenClassification.from_pretrained(str(MODEL_DIR))
    model.to(device)
    model.eval()

    print(f"Loading test set from {TEST_FILE} ...")
    rows = load_test(TEST_FILE)
    print(f"  {len(rows)} records")

    all_true: list[list[str]] = []
    all_pred: list[list[str]] = []

    start = time.time()
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]

        # Pad sequences to the longest in the batch
        max_len = max(len(r["input_ids"]) for r in batch)
        pad_id = tokenizer.pad_token_id or 0

        padded_ids, padded_mask, label_ids = [], [], []
        for r in batch:
            seq_len = len(r["input_ids"])
            pad = max_len - seq_len
            padded_ids.append(r["input_ids"] + [pad_id] * pad)
            padded_mask.append(r["attention_mask"] + [0] * pad)
            label_ids.append(r["labels"] + [-100] * pad)

        input_ids = torch.tensor(padded_ids, dtype=torch.long).to(device)
        attention_mask = torch.tensor(padded_mask, dtype=torch.long).to(device)

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

        preds = torch.argmax(logits, dim=-1).cpu().numpy()

        for pred_seq, true_seq in zip(preds, label_ids):
            pred_labels, true_labels = [], []
            for p, t in zip(pred_seq, true_seq):
                if t == -100:
                    continue
                pred_labels.append(ID2LABEL[int(p)])
                true_labels.append(ID2LABEL[int(t)])
            all_pred.append(pred_labels)
            all_true.append(true_labels)

        if (i // batch_size) % 10 == 0:
            elapsed = time.time() - start
            done = i + len(batch)
            print(f"  {done}/{len(rows)} ({done/len(rows)*100:.1f}%) — {elapsed:.1f}s elapsed")

    elapsed = time.time() - start
    print(f"\nInference done in {elapsed:.1f}s")
    print("\n--- Classification Report ---")
    report_str = classification_report(all_true, all_pred, digits=4)
    print(report_str)

    # Build structured results
    report_dict = classification_report(all_true, all_pred, output_dict=True)
    results = {
        "model": str(MODEL_DIR),
        "test_file": str(TEST_FILE),
        "test_rows": len(rows),
        "inference_seconds": round(elapsed, 1),
        "micro_avg": {
            "precision": round(precision_score(all_true, all_pred), 4),
            "recall": round(recall_score(all_true, all_pred), 4),
            "f1": round(f1_score(all_true, all_pred), 4),
        },
        "per_entity": {},
        "classification_report_text": report_str,
    }
    for label, metrics in report_dict.items():
        if isinstance(metrics, dict) and label not in ("micro avg", "macro avg", "weighted avg"):
            results["per_entity"][label] = {k: round(v, 4) for k, v in metrics.items()}

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    def _native(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        raise TypeError(f"Not serializable: {type(obj)}")

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=_native)

    print(f"\nResults saved to {OUT_FILE}")
    return results


if __name__ == "__main__":
    evaluate()
