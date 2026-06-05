#!/usr/bin/env python3
"""Audit validation-window coordinate alignment against a transcriptome window table."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Tuple


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return open(path, "rt", encoding="utf-8", errors="ignore")


def iter_rows(path: Path) -> Iterable[dict[str, str]]:
    with open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            yield row


def build_window_reference(window_tsv: Path) -> Tuple[set[tuple[str, int, int]], int]:
    keys = set()
    duplicates = 0
    counter: Counter[tuple[str, int, int]] = Counter()
    for row in iter_rows(window_tsv):
        key = (str(row["transcript_id"]), int(row["window_start"]), int(row["window_end"]))
        counter[key] += 1
    for key, count in counter.items():
        keys.add(key)
        if count > 1:
            duplicates += count - 1
    return keys, duplicates


def summarize_matches(
    reference_keys: set[tuple[str, int, int]],
    validation_tsv: Path,
    transcript_col: str,
    start_col: str,
    end_col: str,
    model_name_col: str | None,
    model_name_value: str | None,
) -> Dict[str, object]:
    direct_match = 0
    plus1_match = 0
    duplicate_key_count = 0
    counter: Counter[tuple[str, int, int]] = Counter()
    for row in iter_rows(validation_tsv):
        if model_name_col and model_name_value and str(row.get(model_name_col, "")) != model_name_value:
            continue
        key0 = (str(row[transcript_col]), int(row[start_col]), int(row[end_col]))
        counter[key0] += 1
    unique_total = len(counter)
    for key, count in counter.items():
        if key in reference_keys:
            direct_match += 1
        plus1_key = (key[0], key[1] + 1, key[2])
        if plus1_key in reference_keys:
            plus1_match += 1
        if count > 1:
            duplicate_key_count += count - 1
    selected_offset = 1 if plus1_match > direct_match else 0
    matched = plus1_match if selected_offset == 1 else direct_match
    unmatched = max(unique_total - matched, 0)
    return {
        "validation_tsv": str(validation_tsv),
        "direct_match": int(direct_match),
        "plus1_match": int(plus1_match),
        "selected_offset": int(selected_offset),
        "match_rate": float(matched / unique_total) if unique_total else 0.0,
        "matched_count": int(matched),
        "unmatched_count": int(unmatched),
        "duplicate_key_count": int(duplicate_key_count),
        "total_validation_keys": int(unique_total),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-tsv", required=True)
    parser.add_argument("--observed-tsv", default=None)
    parser.add_argument("--background-tsv", default=None)
    parser.add_argument("--observed-transcript-col", default="transcript_id")
    parser.add_argument("--observed-start-col", default="observed_window_start")
    parser.add_argument("--observed-end-col", default="observed_window_end")
    parser.add_argument("--background-transcript-col", default="transcript_id")
    parser.add_argument("--background-start-col", default="background_window_start")
    parser.add_argument("--background-end-col", default="background_window_end")
    parser.add_argument("--model-name-col", default="model_name")
    parser.add_argument("--model-name-value", default="full_length")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reference_keys, reference_duplicate_key_count = build_window_reference(Path(args.window_tsv))
    report: Dict[str, object] = {
        "window_tsv": str(args.window_tsv),
        "reference_window_count": int(len(reference_keys)),
        "reference_duplicate_key_count": int(reference_duplicate_key_count),
    }
    rows = []
    if args.observed_tsv:
        observed = summarize_matches(
            reference_keys=reference_keys,
            validation_tsv=Path(args.observed_tsv),
            transcript_col=args.observed_transcript_col,
            start_col=args.observed_start_col,
            end_col=args.observed_end_col,
            model_name_col=args.model_name_col,
            model_name_value=args.model_name_value,
        )
        report["observed"] = observed
        rows.append({"subset": "observed", **observed})
    if args.background_tsv:
        background = summarize_matches(
            reference_keys=reference_keys,
            validation_tsv=Path(args.background_tsv),
            transcript_col=args.background_transcript_col,
            start_col=args.background_start_col,
            end_col=args.background_end_col,
            model_name_col=args.model_name_col,
            model_name_value=args.model_name_value,
        )
        report["background"] = background
        rows.append({"subset": "background", **background})

    summary_tsv = out_dir / "coordinate_alignment_report.tsv"
    with open(summary_tsv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["subset"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    report["coordinate_alignment_report_tsv"] = str(summary_tsv)

    summary_json = out_dir / "coordinate_alignment_report.json"
    summary_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(str(summary_tsv))
    print(str(summary_json))


if __name__ == "__main__":
    main()
