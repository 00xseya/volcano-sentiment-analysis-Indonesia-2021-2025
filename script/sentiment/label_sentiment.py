#!/usr/bin/env python3
"""
Labeling sentimen semi-otomatis untuk komentar YouTube gunung Indonesia.

Pipeline:
  1. Dua model sentimen bahasa Indonesia memprediksi probabilitas 3 kelas.
  2. Leksikon/aturan domain menyesuaikan prediksi (doa, duka, humor, hinaan,
     negasi, pertanyaan, dan kata topik bencana).
  3. Hasil digabung, diberi confidence, lalu baris ragu masuk review queue.

Label kanonik: 0 = negatif, 1 = netral, 2 = positif.

Contoh:
  .venv/bin/python script/sentiment/label_sentiment.py \
      --input data/ner_distribution/data_text.csv \
      --output-dir data/sentiment/labeled \
      --batch-size 64

Uji cepat 200 baris:
  ... --limit 200 --output-dir /tmp/sentiment_test
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Dukung pemanggilan langsung maupun sebagai modul.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sentiment_rules import (  # noqa: E402
    ID_TO_LABEL,
    ID_TO_NAME_ID,
    LABEL_TO_ID,
    score_rules,
    strong_label,
)

DEFAULT_MODELS = [
    "w11wo/indonesian-roberta-base-sentiment-classifier",
    "Aardiiiiy/indobertweet-base-Indonesian-sentiment-analysis",
]

CANONICAL_ORDER = ("negative", "neutral", "positive")
POSITIVE_REVIEW_THRESHOLD = 0.65
ACCEPT_THRESHOLD = 0.55
RULE_ALPHA = 0.7


def log(message: str) -> None:
    print(message, flush=True)


def _canonical_index(raw_label: str) -> int:
    raw = raw_label.strip().lower()
    if raw.startswith("neg") or raw in {"label_0", "0"}:
        return 0
    if raw.startswith("neu") or raw in {"label_1", "1"}:
        return 1
    if raw.startswith("pos") or raw in {"label_2", "2"}:
        return 2
    raise ValueError(f"Label model tidak dikenal: {raw_label!r}")


class SentimentModel:
    """Wrapper model HuggingFace dengan output urutan [neg, neu, pos]."""

    def __init__(self, model_name: str, max_length: int = 192, threads: int = 8):
        self.name = model_name
        self.max_length = max_length
        torch.set_num_threads(max(1, threads))
        log(f"[model] memuat {model_name} ...")
        started = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        id2label = self.model.config.id2label
        self.index_map = {i: _canonical_index(id2label[i]) for i in id2label}
        log(f"[model] siap dalam {time.time() - started:.1f}s | label: {id2label}")

    def _prepare_chunks(self, texts: list[str]) -> tuple[list[str], list[int], list[int]]:
        chunks: list[str] = []
        owners: list[int] = []
        lengths: list[int] = []
        budget = self.max_length - 2
        for idx, text in enumerate(texts):
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            if len(ids) <= budget:
                chunks.append(text)
                owners.append(idx)
                lengths.append(max(1, len(ids)))
                continue
            for start in range(0, len(ids), budget):
                piece = self.tokenizer.decode(ids[start:start + budget], skip_special_tokens=True)
                chunks.append(piece)
                owners.append(idx)
                lengths.append(max(1, min(budget, len(ids) - start)))
        return chunks, owners, lengths

    def predict(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        chunks, owners, lengths = self._prepare_chunks(texts)
        n = len(texts)
        summed = np.zeros((n, 3), dtype=np.float64)
        weights = np.zeros(n, dtype=np.float64)

        with torch.inference_mode():
            for start in range(0, len(chunks), batch_size):
                batch = chunks[start:start + batch_size]
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                logits = self.model(**encoded).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                # Susun ulang ke urutan kanonik.
                canonical = np.zeros((len(batch), 3), dtype=np.float64)
                for raw_idx, canonical_idx in self.index_map.items():
                    canonical[:, canonical_idx] = probs[:, raw_idx]
                for j, owner in enumerate(owners[start:start + batch_size]):
                    summed[owner] += canonical[j] * lengths[start + j]
                    weights[owner] += lengths[start + j]

        weights[weights == 0] = 1.0
        return summed / weights[:, None]


def combine_probabilities(
    model_probs: np.ndarray,
    rule_scores: np.ndarray,
    alpha: float = RULE_ALPHA,
) -> np.ndarray:
    """Gabungkan probabilitas model (log-space) dengan rule prior."""
    eps = 1e-6
    logits = np.log(model_probs + eps) + alpha * rule_scores
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def build_rows(
    df: pd.DataFrame,
    probs_per_model: list[np.ndarray],
    alpha: float = RULE_ALPHA,
    text_column: str = "text_cleansed",
) -> pd.DataFrame:
    texts = df[text_column].fillna("").astype(str).tolist()
    ensemble = np.mean(probs_per_model, axis=0)

    rules = [score_rules(t) for t in texts]
    rule_matrix = np.array([r.as_vector() for r in rules], dtype=np.float64)
    combined = combine_probabilities(ensemble, rule_matrix, alpha=alpha)

    strong = [strong_label(t) for t in texts]
    strong_ids = np.array([s[0] if s[0] is not None else -1 for s in strong])
    override_mask = strong_ids >= 0

    labels = combined.argmax(axis=1)
    labels[override_mask] = strong_ids[override_mask]
    confidences = combined.max(axis=1)
    confidences[override_mask] = np.maximum(confidences[override_mask], 0.90)

    model_labels = ensemble.argmax(axis=1)
    model_confidences = ensemble.max(axis=1)
    per_model_labels = [p.argmax(axis=1) for p in probs_per_model]
    per_model_confs = [p.max(axis=1) for p in probs_per_model]
    agreement = np.ones(len(df), dtype=bool)
    for lab in per_model_labels[1:]:
        agreement &= per_model_labels[0] == lab

    rule_labels = np.array([
        (max(range(3), key=lambda i: r.as_vector()[i]) if max(r.as_vector()) > 0 else 1)
        for r in rules
    ])
    rule_margins = np.array([
        sorted(r.as_vector(), reverse=True)[0] - sorted(r.as_vector(), reverse=True)[1]
        for r in rules
    ])
    # Konflik kuat: aturan yakin, model yakin, tapi berbeda.
    rule_conflict = (
        (rule_labels != model_labels)
        & (rule_margins >= 1.5)
        & (model_confidences >= 0.70)
    )

    needs_review = (
        (confidences < POSITIVE_REVIEW_THRESHOLD)
        | (~agreement)
        | rule_conflict
    )
    # Tetap terima jika dua model sepakat dan confidence gabungan cukup tinggi.
    accepted = agreement & (confidences >= ACCEPT_THRESHOLD) & ~rule_conflict
    needs_review = needs_review & ~accepted
    # Aturan presisi tinggi dianggap sudah terverifikasi.
    needs_review[override_mask] = False

    rule_hits = []
    for idx, r in enumerate(rules):
        hits = list(r.hits or [])
        if strong[idx][0] is not None:
            hits.insert(0, f"strong:{strong[idx][1]}")
        rule_hits.append("; ".join(hits))

    result = pd.DataFrame({
        "row_id": (
            df["row_id"].to_numpy() if "row_id" in df.columns else df.index.to_numpy()
        ),
        "video_title": df["video_title"].to_numpy() if "video_title" in df else "",
        "video_year": df["video_year"].to_numpy() if "video_year" in df else "",
        "text": texts,
        "label": labels,
        "label_name": [ID_TO_NAME_ID[int(x)] for x in labels],
        "confidence": np.round(confidences, 4),
        "model_label": model_labels,
        "model_label_name": [ID_TO_NAME_ID[int(x)] for x in model_labels],
        "model_confidence": np.round(model_confidences, 4),
        "model1_label": per_model_labels[0],
        "model1_confidence": np.round(per_model_confs[0], 4),
        "model2_label": per_model_labels[1] if len(per_model_labels) > 1 else -1,
        "model2_confidence": (
            np.round(per_model_confs[1], 4) if len(per_model_confs) > 1 else 0.0
        ),
        "agreement": agreement,
        "rule_label": rule_labels,
        "rule_margin": np.round(rule_margins, 4),
        "rule_hits": rule_hits,
        "needs_review": needs_review,
    })
    result["method"] = np.where(
        override_mask,
        "rule-override",
        np.where(result["rule_hits"].str.len() > 0, "ensemble+rules", "ensemble"),
    )
    return result


def write_outputs(result: pd.DataFrame, output_dir: Path, metadata: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "sentiment_labels.csv"
    jsonl_path = output_dir / "sentiment_labels.jsonl"
    review_path = output_dir.parent / "review" / "review_queue.csv"
    dist_path = output_dir / "label_distribution.csv"
    meta_path = output_dir / "model_metadata.json"

    result.to_csv(csv_path, index=False, encoding="utf-8")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for record in result.to_dict(orient="records"):
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    review = result[result["needs_review"]].sort_values(
        ["confidence", "rule_margin"], ascending=[True, False]
    )
    review_path.parent.mkdir(parents=True, exist_ok=True)
    review.to_csv(review_path, index=False, encoding="utf-8")

    by_year = (
        result.groupby(["video_year", "label_name"])
        .size()
        .unstack(fill_value=0)
    )
    by_year["total"] = by_year.sum(axis=1)
    overall = result["label_name"].value_counts().reindex(
        ["negatif", "netral", "positif"], fill_value=0
    )
    dist = pd.DataFrame({
        "label_name": overall.index,
        "count": overall.values,
        "percentage": np.round(overall.values / len(result) * 100, 2),
    })
    with open(dist_path, "w", encoding="utf-8") as f:
        f.write("# Distribusi keseluruhan\n")
        dist.to_csv(f, index=False)
        f.write("\n# Distribusi per tahun\n")
        by_year.to_csv(f)

    metadata = dict(metadata)
    metadata.update({
        "rows": int(len(result)),
        "review_rows": int(len(review)),
        "label_counts": {k: int(v) for k, v in overall.items()},
        "finished_at": datetime.now(timezone.utc).isoformat(),
    })
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    log(f"[output] {csv_path} ({len(result):,} baris)")
    log(f"[output] {jsonl_path}")
    log(f"[output] {review_path} ({len(review):,} baris review)")
    log(f"[output] {dist_path}")
    log(f"[output] {meta_path}")
    log("\nDistribusi label:")
    log(dist.to_string(index=False))
    log("\nDistribusi per tahun:")
    log(by_year.to_string())


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="data/ner_distribution/data_text.csv",
        help="CSV input dengan kolom text_cleansed (default: data_text.csv)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/sentiment/labeled",
        help="Folder output (default: data/sentiment/labeled)",
    )
    parser.add_argument("--text-column", default="text_cleansed")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 4))
    parser.add_argument("--alpha", type=float, default=RULE_ALPHA,
                        help="Bobot rule prior (default: 0.7)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Batasi jumlah baris untuk uji cepat (0 = semua)")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS),
                        help="Daftar model HuggingFace dipisah koma")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        log(f"[error] Input tidak ditemukan: {input_path}")
        return 1

    log(f"[input] membaca {input_path}")
    df = pd.read_csv(input_path)
    if args.text_column not in df.columns:
        log(f"[error] Kolom {args.text_column!r} tidak ada. Kolom: {list(df.columns)}")
        return 1
    if args.limit and args.limit > 0:
        df = df.head(args.limit).copy()
    log(f"[input] {len(df):,} baris")

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    started = time.time()
    probs_per_model: list[np.ndarray] = []
    for model_name in models:
        model = SentimentModel(model_name, max_length=args.max_length, threads=args.threads)
        t0 = time.time()
        probs = model.predict(df[args.text_column].fillna("").astype(str).tolist(),
                              batch_size=args.batch_size)
        log(f"[model] {model_name} selesai dalam {time.time() - t0:.1f}s")
        probs_per_model.append(probs)

    result = build_rows(df, probs_per_model, alpha=args.alpha,
                        text_column=args.text_column)
    metadata = {
        "pipeline": "sentiment_labeling_v1",
        "input": str(input_path),
        "models": models,
        "label_scheme": {"0": "negatif", "1": "netral", "2": "positif"},
        "canonical_order": list(CANONICAL_ORDER),
        "rule_alpha": args.alpha,
        "accept_threshold": ACCEPT_THRESHOLD,
        "review_threshold": POSITIVE_REVIEW_THRESHOLD,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "git_commit": git_commit(),
        "started_at": datetime.fromtimestamp(started, timezone.utc).isoformat(),
        "duration_seconds": round(time.time() - started, 1),
    }
    write_outputs(result, output_dir, metadata)
    log(f"\n[selesai] total {time.time() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
