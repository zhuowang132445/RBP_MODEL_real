#!/usr/bin/env python3
"""OsDRB1 false-positive and structure-feature audit on fixed best-window predictions."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


HIGH_PAIRED_THRESHOLD = 0.7
RC_KMER_SIZE = 4
RICH_KMER_SIZE = 7


def normalize_seq(seq: str) -> str:
    return str(seq).upper().replace("T", "U")


def load_truth_gene_ids(path: Path) -> set[str]:
    frame = pd.read_csv(path, sep=None, engine="python")
    gene_col = "gene_id" if "gene_id" in frame.columns else frame.columns[0]
    return {str(x) for x in frame[gene_col].dropna().astype(str).tolist()}


def load_positive_urich_kmers(path: Path, top_n: int = 20) -> List[str]:
    frame = pd.read_csv(path, sep="\t")
    pos = frame[frame["direction"].astype(str) == "positive"].copy()
    if pos.empty:
        return []
    pos["kmer"] = pos["kmer"].astype(str).map(normalize_seq)
    pos["uc_fraction"] = pos["kmer"].map(lambda x: (x.count("U") + x.count("C")) / max(len(x), 1))
    pos["u_fraction"] = pos["kmer"].map(lambda x: x.count("U") / max(len(x), 1))
    pos = pos.sort_values(["predicted_zscore", "rank"], ascending=[False, True])
    filtered = pos[(pos["uc_fraction"] >= 0.71) & (pos["u_fraction"] >= 0.43)]
    if filtered.empty:
        filtered = pos.head(top_n)
    return filtered["kmer"].drop_duplicates().head(top_n).tolist()


def reverse_complement_rna(seq: str) -> str:
    table = str.maketrans({"A": "U", "U": "A", "C": "G", "G": "C"})
    return normalize_seq(seq).translate(table)[::-1]


def compute_sequence_features(seq: str, urich_kmers: Sequence[str]) -> Dict[str, float]:
    seq = normalize_seq(seq)
    length = len(seq)
    if length == 0:
        return {
            "U_content": math.nan,
            "C_content": math.nan,
            "UC_content": math.nan,
            "AU_content": math.nan,
            "max_homopolymer_U_len": math.nan,
            "shannon_entropy": math.nan,
            "low_complexity_score": math.nan,
            "UUUUUUU_count": math.nan,
            "top_Urich_kmer_density": math.nan,
            "reverse_complement_kmer_pair_density": math.nan,
        }

    counts = {base: seq.count(base) for base in "ACGU"}
    probs = np.array([counts[base] / length for base in "ACGU"], dtype=np.float64)
    probs = probs[probs > 0]
    entropy = float(-(probs * np.log2(probs)).sum()) if len(probs) else math.nan
    low_complexity = float(1.0 - min(entropy / 2.0, 1.0)) if np.isfinite(entropy) else math.nan

    longest_u = 0
    current_u = 0
    for base in seq:
        if base == "U":
            current_u += 1
            longest_u = max(longest_u, current_u)
        else:
            current_u = 0

    motif_hits = 0
    total_k = max(length - RICH_KMER_SIZE + 1, 0)
    if total_k > 0 and urich_kmers:
        urich_set = set(urich_kmers)
        for i in range(total_k):
            if seq[i : i + RICH_KMER_SIZE] in urich_set:
                motif_hits += 1
        urich_density = motif_hits / total_k
    else:
        urich_density = math.nan

    total_rc_k = max(length - RC_KMER_SIZE + 1, 0)
    rc_density = math.nan
    if total_rc_k > 0:
        kmers = [seq[i : i + RC_KMER_SIZE] for i in range(total_rc_k)]
        kmer_set = set(kmers)
        rc_hits = sum(1 for kmer in kmers if reverse_complement_rna(kmer) in kmer_set)
        rc_density = rc_hits / total_rc_k

    return {
        "U_content": counts["U"] / length,
        "C_content": counts["C"] / length,
        "UC_content": (counts["U"] + counts["C"]) / length,
        "AU_content": (counts["A"] + counts["U"]) / length,
        "max_homopolymer_U_len": float(longest_u),
        "shannon_entropy": entropy,
        "low_complexity_score": low_complexity,
        "UUUUUUU_count": float(sum(1 for i in range(max(length - 7 + 1, 0)) if seq[i : i + 7] == "UUUUUUU")),
        "top_Urich_kmer_density": urich_density,
        "reverse_complement_kmer_pair_density": rc_density,
    }


def run_lengths(mask: np.ndarray) -> List[int]:
    runs: List[int] = []
    current = 0
    for flag in mask.astype(bool).tolist():
        if flag:
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return runs


def max_sliding_mean(values: np.ndarray, width: int) -> float:
    if values.size == 0:
        return math.nan
    if values.size < width:
        return float(np.nanmean(values))
    kernel = np.ones(width, dtype=np.float64) / float(width)
    conv = np.convolve(values.astype(np.float64), kernel, mode="valid")
    return float(np.nanmax(conv))


def compute_structure_features(seq: str, paired: np.ndarray) -> Dict[str, float]:
    paired = np.asarray(paired, dtype=np.float32)
    paired = paired[np.isfinite(paired)]
    if paired.size == 0:
        return {
            "longest_high_paired_run": math.nan,
            "high_paired_run_count": math.nan,
            "paired_run_density": math.nan,
            "local_41nt_paired_mean": math.nan,
            "local_81nt_paired_mean": math.nan,
            "AU_rich_paired_density": math.nan,
            "U_rich_paired_density": math.nan,
        }
    high = paired >= HIGH_PAIRED_THRESHOLD
    runs = run_lengths(high)
    total_windows = max(len(seq) - RICH_KMER_SIZE + 1, 0)
    au_rich_hits = 0
    u_rich_hits = 0
    seq = normalize_seq(seq)
    if total_windows > 0:
        for i in range(total_windows):
            sub = seq[i : i + RICH_KMER_SIZE]
            sub_paired = paired[i : i + RICH_KMER_SIZE]
            if len(sub_paired) < RICH_KMER_SIZE:
                continue
            au_fraction = (sub.count("A") + sub.count("U")) / RICH_KMER_SIZE
            u_fraction = sub.count("U") / RICH_KMER_SIZE
            mean_paired = float(np.nanmean(sub_paired))
            if au_fraction >= 6.0 / 7.0 and mean_paired >= HIGH_PAIRED_THRESHOLD:
                au_rich_hits += 1
            if u_fraction >= 5.0 / 7.0 and mean_paired >= HIGH_PAIRED_THRESHOLD:
                u_rich_hits += 1
    return {
        "longest_high_paired_run": float(max(runs) if runs else 0),
        "high_paired_run_count": float(len(runs)),
        "paired_run_density": float(high.mean()),
        "local_41nt_paired_mean": max_sliding_mean(paired, 41),
        "local_81nt_paired_mean": max_sliding_mean(paired, 81),
        "AU_rich_paired_density": float(au_rich_hits / total_windows) if total_windows > 0 else math.nan,
        "U_rich_paired_density": float(u_rich_hits / total_windows) if total_windows > 0 else math.nan,
    }


def build_window_row_index(window_table: Path, target_keys: set[Tuple[str, int, int]]) -> Tuple[Dict[Tuple[str, int, int], int], int]:
    matched: Dict[Tuple[str, int, int], int] = {}
    duplicate_count = 0
    row_offset = 0
    for chunk in pd.read_csv(
        window_table,
        sep="\t",
        compression="gzip",
        usecols=["transcript_id", "window_start", "window_end"],
        chunksize=200000,
    ):
        for row in chunk.itertuples(index=False):
            key = (str(row.transcript_id), int(row.window_start), int(row.window_end))
            if key in target_keys:
                if key in matched:
                    duplicate_count += 1
                else:
                    matched[key] = row_offset
            row_offset += 1
    return matched, duplicate_count


def group_summary(frame: pd.DataFrame, numeric_cols: Sequence[str]) -> pd.DataFrame:
    rows = []
    for group_name, sub in frame.groupby("audit_group", sort=False):
        row = {"audit_group": group_name, "n_genes": int(len(sub))}
        for col in numeric_cols:
            values = pd.to_numeric(sub[col], errors="coerce")
            row[f"{col}_mean"] = float(values.mean()) if len(values.dropna()) else math.nan
            row[f"{col}_median"] = float(values.median()) if len(values.dropna()) else math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gene-score-table", required=True)
    parser.add_argument("--truth-gene-list", required=True)
    parser.add_argument("--window-table", required=True)
    parser.add_argument("--structure-npy", required=True)
    parser.add_argument("--motif-top-kmers", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--top-k", type=int, default=1000)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gene_scores = pd.read_csv(args.gene_score_table, sep="\t")
    gene_scores["gene_id"] = gene_scores["gene_id"].astype(str)
    gene_scores["transcript_id"] = gene_scores["transcript_id"].astype(str)
    gene_scores = gene_scores.sort_values(["top3_mean_score", "gene_id"], ascending=[False, True], na_position="last").reset_index(drop=True)
    gene_scores["rank_top3_mean_score"] = np.arange(1, len(gene_scores) + 1, dtype=np.int64)

    truth_gene_ids = load_truth_gene_ids(Path(args.truth_gene_list))
    comparable_truth = set(gene_scores["gene_id"]) & truth_gene_ids
    top = gene_scores.head(int(args.top_k)).copy()
    top_truth = set(top.loc[top["gene_id"].isin(comparable_truth), "gene_id"].tolist())
    top_false = set(top.loc[~top["gene_id"].isin(comparable_truth), "gene_id"].tolist())
    outside_truth = comparable_truth - set(top["gene_id"])

    focus = gene_scores[
        gene_scores["gene_id"].isin(top_truth | top_false | outside_truth)
    ].copy()
    focus["audit_group"] = "unassigned"
    focus.loc[focus["gene_id"].isin(top_truth), "audit_group"] = "top1000_true_positive"
    focus.loc[focus["gene_id"].isin(top_false), "audit_group"] = "top1000_false_positive"
    focus.loc[focus["gene_id"].isin(outside_truth), "audit_group"] = "truth_outside_top1000"

    urich_kmers = load_positive_urich_kmers(Path(args.motif_top_kmers))

    target_keys = {
        (str(row.transcript_id), int(row.best_window_start), int(row.best_window_end))
        for row in focus.itertuples(index=False)
    }
    row_index, duplicate_count = build_window_row_index(Path(args.window_table), target_keys)
    paired_matrix = np.load(args.structure_npy, mmap_mode="r")

    seq_feature_rows = []
    structure_feature_rows = []
    missing_keys = 0
    for row in focus.itertuples(index=False):
        key = (str(row.transcript_id), int(row.best_window_start), int(row.best_window_end))
        seq = normalize_seq(row.best_window_seq)
        seq_features = compute_sequence_features(seq, urich_kmers)
        if key in row_index:
            paired = np.asarray(paired_matrix[row_index[key]], dtype=np.float32)
            paired = paired[: len(seq)]
            struct_features = compute_structure_features(seq, paired)
        else:
            missing_keys += 1
            struct_features = {
                "longest_high_paired_run": math.nan,
                "high_paired_run_count": math.nan,
                "paired_run_density": math.nan,
                "local_41nt_paired_mean": math.nan,
                "local_81nt_paired_mean": math.nan,
                "AU_rich_paired_density": math.nan,
                "U_rich_paired_density": math.nan,
            }
        seq_feature_rows.append(seq_features)
        structure_feature_rows.append(struct_features)

    seq_df = pd.DataFrame(seq_feature_rows)
    struct_df = pd.DataFrame(structure_feature_rows)
    merged = pd.concat([focus.reset_index(drop=True), seq_df, struct_df], axis=1)
    merged["region_annotation"] = "NA"
    merged["repeat_annotation"] = "NA"
    merged["utr_annotation"] = "NA"
    merged["intron_annotation"] = "NA"
    merged["mirna_annotation"] = "NA"

    false_positive_cols = [
        "audit_group",
        "rank_top3_mean_score",
        "gene_id",
        "transcript_id",
        "n_windows",
        "best_window_start",
        "best_window_end",
        "best_window_seq",
        "best_window_score",
        "top3_mean_score",
        "motif_score",
        "structure_score",
        "base_score",
        "paired_probability_mean",
        "fraction_high_paired",
        "U_content",
        "C_content",
        "UC_content",
        "AU_content",
        "max_homopolymer_U_len",
        "shannon_entropy",
        "low_complexity_score",
        "UUUUUUU_count",
        "top_Urich_kmer_density",
        "reverse_complement_kmer_pair_density",
        "longest_high_paired_run",
        "high_paired_run_count",
        "paired_run_density",
        "local_41nt_paired_mean",
        "local_81nt_paired_mean",
        "AU_rich_paired_density",
        "U_rich_paired_density",
        "region_annotation",
        "repeat_annotation",
        "utr_annotation",
        "intron_annotation",
        "mirna_annotation",
    ]
    merged[false_positive_cols].to_csv(out_dir / "false_positive_audit.tsv", sep="\t", index=False)

    structure_cols = [
        "audit_group",
        "rank_top3_mean_score",
        "gene_id",
        "transcript_id",
        "best_window_start",
        "best_window_end",
        "paired_probability_mean",
        "fraction_high_paired",
        "reverse_complement_kmer_pair_density",
        "longest_high_paired_run",
        "high_paired_run_count",
        "paired_run_density",
        "local_41nt_paired_mean",
        "local_81nt_paired_mean",
        "AU_rich_paired_density",
        "U_rich_paired_density",
    ]
    merged[structure_cols].to_csv(out_dir / "osdrb1_structure_feature_audit.tsv", sep="\t", index=False)

    numeric_cols = [
        "n_windows",
        "top3_mean_score",
        "motif_score",
        "structure_score",
        "base_score",
        "paired_probability_mean",
        "fraction_high_paired",
        "U_content",
        "C_content",
        "UC_content",
        "AU_content",
        "max_homopolymer_U_len",
        "shannon_entropy",
        "low_complexity_score",
        "UUUUUUU_count",
        "top_Urich_kmer_density",
        "reverse_complement_kmer_pair_density",
        "longest_high_paired_run",
        "high_paired_run_count",
        "paired_run_density",
        "local_41nt_paired_mean",
        "local_81nt_paired_mean",
        "AU_rich_paired_density",
        "U_rich_paired_density",
    ]
    group_summary(merged, numeric_cols).to_csv(out_dir / "false_positive_audit_group_summary.tsv", sep="\t", index=False)

    report = {
        "gene_score_table": str(args.gene_score_table),
        "truth_gene_list": str(args.truth_gene_list),
        "window_table": str(args.window_table),
        "structure_npy": str(args.structure_npy),
        "top_k": int(args.top_k),
        "n_ranked_genes": int(len(gene_scores)),
        "n_comparable_truth": int(len(comparable_truth)),
        "n_top1000_true_positive": int(len(top_truth)),
        "n_top1000_false_positive": int(len(top_false)),
        "n_truth_outside_top1000": int(len(outside_truth)),
        "n_focus_genes": int(len(merged)),
        "matched_window_keys": int(len(row_index)),
        "missing_window_keys": int(missing_keys),
        "duplicate_window_keys": int(duplicate_count),
        "high_paired_threshold": HIGH_PAIRED_THRESHOLD,
        "reverse_complement_kmer_size": RC_KMER_SIZE,
        "rich_kmer_size": RICH_KMER_SIZE,
        "selected_urich_kmers": urich_kmers,
        "metric_notes": {
            "low_complexity_score": "1 - shannon_entropy / 2.0 on A/C/G/U alphabet",
            "paired_run_density": "fraction of positions with paired probability >= threshold",
            "local_41nt_paired_mean": "maximum sliding-window mean paired probability over 41 nt",
            "local_81nt_paired_mean": "maximum sliding-window mean paired probability over 81 nt",
            "reverse_complement_kmer_pair_density": "fraction of 4-mers whose RNA reverse complement also appears in the same best window",
            "AU_rich_paired_density": "fraction of 7-mers with AU fraction >= 6/7 and mean paired probability >= threshold",
            "U_rich_paired_density": "fraction of 7-mers with U fraction >= 5/7 and mean paired probability >= threshold",
        },
    }
    (out_dir / "false_positive_audit_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
