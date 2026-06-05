#!/usr/bin/env python3
"""Audit gene-level false positives for an OsDRB1-style transcriptome prediction run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load_truth_gene_ids(path: Path) -> set[str]:
    frame = pd.read_csv(path, sep=None, engine="python")
    for column in ["gene_id", "target_gene_id", "truth_gene_id", "gene"]:
        if column in frame.columns:
            return {str(x) for x in frame[column].dropna().astype(str).tolist()}
    if len(frame.columns) == 1:
        return {str(x) for x in frame.iloc[:, 0].dropna().astype(str).tolist()}
    raise ValueError(f"cannot infer truth gene id column from {path}")


def pick_ranking_score_column(frame: pd.DataFrame, preferred: str | None) -> str:
    if preferred and preferred in frame.columns:
        return preferred
    for candidate in ["ranking_score", "max_score", "top3_mean_score", "top5_mean_score", "window_count_adjusted_score"]:
        if candidate in frame.columns:
            return candidate
    raise ValueError("no usable ranking score column found in gene score table")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gene-scores", required=True)
    parser.add_argument("--truth-gene-list", required=True)
    parser.add_argument("--rbp-id", required=True)
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument("--ranking-score-col", default=None)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gene_scores = pd.read_csv(args.gene_scores, sep="\t")
    gene_scores = gene_scores[gene_scores["rbp_id"].astype(str) == str(args.rbp_id)].copy()
    if gene_scores.empty:
        raise ValueError(f"no rows for rbp_id={args.rbp_id} in {args.gene_scores}")
    truth_gene_ids = load_truth_gene_ids(Path(args.truth_gene_list))
    ranking_score_col = pick_ranking_score_column(gene_scores, args.ranking_score_col)

    gene_scores = gene_scores.sort_values([ranking_score_col, "gene_id"], ascending=[False, True], na_position="last").reset_index(drop=True)
    gene_scores["audit_rank"] = range(1, len(gene_scores) + 1)
    gene_scores["is_truth_target"] = gene_scores["gene_id"].astype(str).isin(truth_gene_ids)

    top = gene_scores.head(int(args.top_k)).copy()
    top_tp = top[top["is_truth_target"]].copy()
    top_tp["audit_group"] = "top1000_true_positive"
    top_fp = top[~top["is_truth_target"]].copy()
    top_fp["audit_group"] = "top1000_false_positive"
    outside = gene_scores[(gene_scores["is_truth_target"]) & (gene_scores["audit_rank"] > int(args.top_k))].copy()
    outside["audit_group"] = "true_target_outside_top1000"

    audit = pd.concat([top_tp, top_fp, outside], ignore_index=True)
    audit["motif_score"] = audit.get("best_window_motif_match_score")
    audit["structure_score"] = audit.get("best_window_structure_match_score")
    audit["base_score"] = audit.get("best_window_base_score")
    audit["paired_probability_mean"] = audit.get("best_window_paired_probability_mean")
    audit["fraction_high_paired"] = audit.get("best_window_fraction_high_paired")
    audit["region_annotation"] = ""
    audit["repeat_annotation"] = ""
    audit["utr_annotation"] = ""
    audit["intron_annotation"] = ""
    audit["mirna_annotation"] = ""
    audit = audit[
        [
            "audit_group",
            "audit_rank",
            "rbp_id",
            "gene_id",
            ranking_score_col,
            "motif_score",
            "structure_score",
            "base_score",
            "n_windows",
            "best_window_seq",
            "best_window_start",
            "best_window_end",
            "paired_probability_mean",
            "fraction_high_paired",
            "region_annotation",
            "repeat_annotation",
            "utr_annotation",
            "intron_annotation",
            "mirna_annotation",
        ]
    ].rename(columns={ranking_score_col: "ranking_score"})

    audit_tsv = out_dir / "osdrb1_false_positive_audit.tsv"
    audit.to_csv(audit_tsv, sep="\t", index=False)

    summary = {
        "rbp_id": str(args.rbp_id),
        "gene_scores_tsv": str(args.gene_scores),
        "truth_gene_list": str(args.truth_gene_list),
        "ranking_score_col": ranking_score_col,
        "top_k": int(args.top_k),
        "n_gene_scores": int(len(gene_scores)),
        "n_truth_genes_total": int(len(truth_gene_ids)),
        "top_k_true_positive_count": int(len(top_tp)),
        "top_k_false_positive_count": int(len(top_fp)),
        "true_targets_outside_top_k_count": int(len(outside)),
        "audit_tsv": str(audit_tsv),
    }
    summary_json = out_dir / "osdrb1_false_positive_audit_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(str(audit_tsv))
    print(str(summary_json))


if __name__ == "__main__":
    main()
