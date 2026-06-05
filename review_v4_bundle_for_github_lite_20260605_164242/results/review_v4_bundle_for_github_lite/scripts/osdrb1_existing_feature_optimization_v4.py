#!/usr/bin/env python3
"""OsDRB1 existing-feature optimization without retraining deep models."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from scipy.stats import hypergeom
except Exception:  # pragma: no cover
    hypergeom = None


CANDIDATE_SCORES = [
    "base_score",
    "inverted_base_score",
    "motif_only_score",
    "structure_only_score",
    "motif_structure_score",
    "base_motif_structure_score",
    "inverted_base_motif_structure_score",
]

CANDIDATE_AGGREGATIONS = [
    "max_score",
    "top3_mean_score",
    "top5_mean_score",
    "top10_mean_score",
    "top5_percent_mean_score",
]

TOP_KS = [50, 100, 200, 500, 1000]
HIGH_PAIRED_THRESHOLD = 0.7
RC_KMER_SIZE = 4
RICH_KMER_SIZE = 7
TOP_WINDOW_CAP = 10

FEATURE_COLUMNS = [
    "max_score",
    "top3_mean_score",
    "top5_mean_score",
    "top10_mean_score",
    "top5_percent_mean_score",
    "max_base_score",
    "min_base_score",
    "mean_top3_inverted_base_score",
    "mean_top5_inverted_base_score",
    "top3_motif_score",
    "top5_motif_score",
    "max_motif_score",
    "motif_positive_fraction_top3_mean",
    "top3_structure_score",
    "top5_structure_score",
    "max_structure_score",
    "top3_paired_probability_mean",
    "top3_fraction_high_paired",
    "local_41nt_paired_mean_top3",
    "local_81nt_paired_mean_top3",
    "longest_high_paired_run_max",
    "paired_run_density_top3",
    "reverse_complement_kmer_pair_density_top3",
    "AU_rich_paired_density_top3",
    "U_rich_paired_density_top3",
    "U_content_top3",
    "UC_content_top3",
    "AU_content_top3",
    "max_U_homopolymer_len_max",
    "shannon_entropy_top3",
    "low_complexity_score_top3",
    "UUUUUUU_count_max",
    "top_Urich_kmer_density_top3",
    "n_windows",
    "log1p_n_windows",
    "n_windows_bin_code",
    "n_high_score_windows_0.9",
    "n_high_score_windows_0.95",
    "fraction_high_score_windows_0.9",
    "fraction_high_score_windows_0.95",
]

B_BRANCH_FEATURES = [
    "local_41nt_paired_mean_top3",
    "local_81nt_paired_mean_top3",
    "reverse_complement_kmer_pair_density_top3",
    "AU_rich_paired_density_top3",
    "U_rich_paired_density_top3",
    "paired_run_density_top3",
]


def safe_hypergeom_sf(k_minus_one: int, m: int, n: int, n_draws: int) -> float:
    if hypergeom is not None:
        return float(hypergeom.sf(k_minus_one, m, n, n_draws))
    denom = math.comb(m, n_draws)
    total = 0.0
    for k in range(max(0, k_minus_one + 1), min(n, n_draws) + 1):
        total += (math.comb(n, k) * math.comb(m - n, n_draws - k)) / denom
    return float(min(max(total, 0.0), 1.0))


def load_truth_gene_ids(path: Path) -> set[str]:
    frame = pd.read_csv(path, sep=None, engine="python")
    for column in ["gene_id", "target_gene_id", "truth_gene_id", "gene"]:
        if column in frame.columns:
            return {str(x) for x in frame[column].dropna().astype(str).tolist()}
    if len(frame.columns) == 1:
        return {str(x) for x in frame.iloc[:, 0].dropna().astype(str).tolist()}
    raise ValueError(f"cannot infer truth gene id column from {path}")


def normalize_seq(seq: str) -> str:
    return str(seq).upper().replace("T", "U")


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
    probs = np.asarray([counts[base] / length for base in "ACGU"], dtype=np.float64)
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

    total_rich = max(length - RICH_KMER_SIZE + 1, 0)
    top_urich_density = math.nan
    if total_rich > 0 and urich_kmers:
        urich_set = set(urich_kmers)
        hits = sum(1 for i in range(total_rich) if seq[i : i + RICH_KMER_SIZE] in urich_set)
        top_urich_density = hits / total_rich

    total_rc = max(length - RC_KMER_SIZE + 1, 0)
    rc_density = math.nan
    if total_rc > 0:
        kmers = [seq[i : i + RC_KMER_SIZE] for i in range(total_rc)]
        kmer_set = set(kmers)
        rc_hits = sum(1 for kmer in kmers if reverse_complement_rna(kmer) in kmer_set)
        rc_density = rc_hits / total_rc

    return {
        "U_content": counts["U"] / length,
        "C_content": counts["C"] / length,
        "UC_content": (counts["U"] + counts["C"]) / length,
        "AU_content": (counts["A"] + counts["U"]) / length,
        "max_homopolymer_U_len": float(longest_u),
        "shannon_entropy": entropy,
        "low_complexity_score": low_complexity,
        "UUUUUUU_count": float(sum(1 for i in range(max(length - 6, 0)) if seq[i : i + 7] == "UUUUUUU")),
        "top_Urich_kmer_density": top_urich_density,
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
    seq = normalize_seq(seq)
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
    au_hits = 0
    u_hits = 0
    if total_windows > 0:
        for i in range(total_windows):
            sub = seq[i : i + RICH_KMER_SIZE]
            sub_paired = paired[i : i + RICH_KMER_SIZE]
            if len(sub_paired) < RICH_KMER_SIZE:
                continue
            mean_paired = float(np.nanmean(sub_paired))
            au_fraction = (sub.count("A") + sub.count("U")) / RICH_KMER_SIZE
            u_fraction = sub.count("U") / RICH_KMER_SIZE
            if au_fraction >= 6.0 / 7.0 and mean_paired >= HIGH_PAIRED_THRESHOLD:
                au_hits += 1
            if u_fraction >= 5.0 / 7.0 and mean_paired >= HIGH_PAIRED_THRESHOLD:
                u_hits += 1

    return {
        "longest_high_paired_run": float(max(runs) if runs else 0),
        "high_paired_run_count": float(len(runs)),
        "paired_run_density": float(high.mean()),
        "local_41nt_paired_mean": max_sliding_mean(paired, 41),
        "local_81nt_paired_mean": max_sliding_mean(paired, 81),
        "AU_rich_paired_density": float(au_hits / total_windows) if total_windows > 0 else math.nan,
        "U_rich_paired_density": float(u_hits / total_windows) if total_windows > 0 else math.nan,
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


def evaluate_ranked_frame(
    frame: pd.DataFrame,
    score_col: str,
    truth_gene_ids: set[str],
    split_name: str,
    seed: int,
    label: str,
    top_ks: Iterable[int],
) -> pd.DataFrame:
    ranked = frame.sort_values([score_col, "gene_id"], ascending=[False, True], na_position="last").reset_index(drop=True)
    universe = {str(x) for x in ranked["gene_id"].astype(str).tolist()}
    comparable_truth = universe & truth_gene_ids
    m = len(universe)
    n = len(comparable_truth)
    rows = []
    for top_k in top_ks:
        subset = ranked.head(min(int(top_k), len(ranked)))
        hits = int(subset["gene_id"].astype(str).isin(comparable_truth).sum())
        expected = (len(subset) * n / m) if m else float("nan")
        precision = (hits / len(subset)) if len(subset) else float("nan")
        recall = (hits / n) if n else float("nan")
        fold = (hits / expected) if expected and np.isfinite(expected) and expected > 0 else float("nan")
        pvalue = safe_hypergeom_sf(hits - 1, m, n, len(subset)) if m and n and len(subset) else float("nan")
        rows.append(
            {
                "seed": int(seed),
                "label": label,
                "split_name": split_name,
                "top_k": int(top_k),
                "universe_genes": int(m),
                "truth_genes": int(n),
                "overlap": int(hits),
                "precision": float(precision),
                "recall": float(recall),
                "expected_random": float(expected),
                "fold_enrichment": float(fold),
                "hypergeom_pvalue": float(pvalue),
            }
        )
    return pd.DataFrame(rows)


def select_candidate_by_objective(calibration_rows: pd.DataFrame, objective: str) -> pd.Series:
    records: List[Dict[str, float]] = []
    for (score_name, aggregation), sub in calibration_rows.groupby(["score_name", "aggregation"], sort=False):
        row = {"score_name": score_name, "aggregation": aggregation}
        for top_k in [200, 500, 1000]:
            sub_top = sub[sub["top_k"] == top_k].iloc[0]
            row[f"top{top_k}_overlap"] = float(sub_top["overlap"])
            row[f"top{top_k}_precision"] = float(sub_top["precision"])
            row[f"top{top_k}_recall"] = float(sub_top["recall"])
            row[f"top{top_k}_fold"] = float(sub_top["fold_enrichment"])
            row[f"top{top_k}_hypergeom"] = float(sub_top["hypergeom_pvalue"])
        records.append(row)
    wide = pd.DataFrame(records)
    wide["weighted_top200_500_1000"] = (
        0.5 * -np.log10(np.clip(wide["top200_hypergeom"].to_numpy(dtype=float), 1e-300, 1.0))
        + 0.3 * -np.log10(np.clip(wide["top500_hypergeom"].to_numpy(dtype=float), 1e-300, 1.0))
        + 0.2 * -np.log10(np.clip(wide["top1000_hypergeom"].to_numpy(dtype=float), 1e-300, 1.0))
    )

    if objective == "top200_overlap":
        sort_cols = ["top200_overlap", "top200_fold", "top200_hypergeom", "top500_overlap", "top1000_overlap"]
        ascending = [False, False, True, False, False]
    elif objective == "top200_hypergeom":
        sort_cols = ["top200_hypergeom", "top200_overlap", "top200_fold", "top500_hypergeom", "top1000_hypergeom"]
        ascending = [True, False, False, True, True]
    elif objective == "top200_fold":
        sort_cols = ["top200_fold", "top200_overlap", "top200_hypergeom", "top500_fold", "top1000_fold"]
        ascending = [False, False, True, False, False]
    elif objective == "weighted_top200_500_1000":
        sort_cols = ["weighted_top200_500_1000", "top200_overlap", "top500_overlap", "top1000_overlap"]
        ascending = [False, False, False, False]
    elif objective == "top500_hypergeom":
        sort_cols = ["top500_hypergeom", "top500_overlap", "top500_fold", "top200_hypergeom", "top1000_hypergeom"]
        ascending = [True, False, False, True, True]
    elif objective == "top1000_hypergeom":
        sort_cols = ["top1000_hypergeom", "top1000_overlap", "top1000_fold", "top500_hypergeom", "top200_hypergeom"]
        ascending = [True, False, False, True, True]
    else:
        raise ValueError(f"unknown objective: {objective}")

    wide = wide.sort_values(sort_cols, ascending=ascending, na_position="last").reset_index(drop=True)
    return wide.iloc[0]


def merge_top_windows(existing: List[Dict[str, object]], new_items: List[Dict[str, object]], cap: int = TOP_WINDOW_CAP) -> List[Dict[str, object]]:
    combined = existing + new_items
    combined.sort(
        key=lambda item: (
            -float(item["rank_score"]),
            str(item["transcript_id"]),
            int(item["window_start"]),
            int(item["window_end"]),
        )
    )
    return combined[:cap]


def nanmean_list(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(arr)) if np.isfinite(arr).any() else math.nan


def nanmax_list(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float(np.nanmax(arr)) if np.isfinite(arr).any() else math.nan


def percentile_rank(values: pd.Series) -> pd.Series:
    return values.rank(method="average", pct=True)


def build_existing_feature_gene_table(
    gene_table_path: Path,
    window_scores_path: Path,
    window_table_path: Path,
    structure_npy_path: Path,
    motif_top_kmers_path: Path,
    out_path: Path,
    chunksize: int,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    gene_table = pd.read_csv(gene_table_path, sep="\t")
    gene_table["gene_id"] = gene_table["gene_id"].astype(str)
    gene_table["transcript_id"] = gene_table["transcript_id"].astype(str)

    usecols = [
        "gene_id",
        "transcript_id",
        "window_start",
        "window_end",
        "rna_seq",
        "base_score",
        "inverted_base_score",
        "inverted_base_motif_structure_score",
        "motif_match_score",
        "motif_positive_fraction",
        "structure_match_raw",
        "paired_probability_mean",
        "fraction_high_paired",
    ]

    agg_map: Dict[str, Dict[str, float]] = {}
    top_windows: Dict[str, List[Dict[str, object]]] = {}

    for chunk in pd.read_csv(window_scores_path, sep="\t", compression="gzip", usecols=usecols, chunksize=chunksize):
        chunk["gene_id"] = chunk["gene_id"].astype(str)
        chunk["transcript_id"] = chunk["transcript_id"].astype(str)
        chunk["score_ge_0_9"] = chunk["inverted_base_motif_structure_score"] >= 0.9
        chunk["score_ge_0_95"] = chunk["inverted_base_motif_structure_score"] >= 0.95

        grouped = (
            chunk.groupby("gene_id", sort=False)
            .agg(
                max_base_score=("base_score", "max"),
                min_base_score=("base_score", "min"),
                max_motif_score=("motif_match_score", "max"),
                max_structure_score=("structure_match_raw", "max"),
                n_high_score_windows_0_9=("score_ge_0_9", "sum"),
                n_high_score_windows_0_95=("score_ge_0_95", "sum"),
            )
            .reset_index()
        )
        for row in grouped.itertuples(index=False):
            gene_id = str(row.gene_id)
            current = agg_map.get(gene_id)
            if current is None:
                agg_map[gene_id] = {
                    "max_base_score": float(row.max_base_score),
                    "min_base_score": float(row.min_base_score),
                    "max_motif_score": float(row.max_motif_score),
                    "max_structure_score": float(row.max_structure_score),
                    "n_high_score_windows_0.9": int(row.n_high_score_windows_0_9),
                    "n_high_score_windows_0.95": int(row.n_high_score_windows_0_95),
                }
            else:
                current["max_base_score"] = float(np.nanmax([current["max_base_score"], row.max_base_score]))
                current["min_base_score"] = float(np.nanmin([current["min_base_score"], row.min_base_score]))
                current["max_motif_score"] = float(np.nanmax([current["max_motif_score"], row.max_motif_score]))
                current["max_structure_score"] = float(np.nanmax([current["max_structure_score"], row.max_structure_score]))
                current["n_high_score_windows_0.9"] += int(row.n_high_score_windows_0_9)
                current["n_high_score_windows_0.95"] += int(row.n_high_score_windows_0_95)

        top_chunk = (
            chunk.sort_values(
                ["gene_id", "inverted_base_motif_structure_score", "transcript_id", "window_start", "window_end"],
                ascending=[True, False, True, True, True],
            )
            .groupby("gene_id", sort=False)
            .head(TOP_WINDOW_CAP)
        )
        for gene_id, sub in top_chunk.groupby("gene_id", sort=False):
            new_items = []
            for row in sub.itertuples(index=False):
                new_items.append(
                    {
                        "transcript_id": str(row.transcript_id),
                        "window_start": int(row.window_start),
                        "window_end": int(row.window_end),
                        "rna_seq": normalize_seq(row.rna_seq),
                        "rank_score": float(row.inverted_base_motif_structure_score),
                        "base_score": float(row.base_score),
                        "inverted_base_score": float(row.inverted_base_score),
                        "motif_match_score": float(row.motif_match_score),
                        "motif_positive_fraction": float(row.motif_positive_fraction),
                        "structure_match_raw": float(row.structure_match_raw),
                        "paired_probability_mean": float(row.paired_probability_mean),
                        "fraction_high_paired": float(row.fraction_high_paired),
                    }
                )
            top_windows[str(gene_id)] = merge_top_windows(top_windows.get(str(gene_id), []), new_items)

    urich_kmers = load_positive_urich_kmers(motif_top_kmers_path)
    target_keys = {
        (window["transcript_id"], window["window_start"], window["window_end"])
        for windows in top_windows.values()
        for window in windows[:3]
    }
    row_index, duplicate_count = build_window_row_index(window_table_path, target_keys)
    paired_matrix = np.load(structure_npy_path, mmap_mode="r")

    per_window_extra: Dict[Tuple[str, int, int], Dict[str, float]] = {}
    missing_structure_keys = 0
    for windows in top_windows.values():
        for window in windows[:3]:
            key = (window["transcript_id"], window["window_start"], window["window_end"])
            if key in per_window_extra:
                continue
            seq_features = compute_sequence_features(window["rna_seq"], urich_kmers)
            struct_features: Dict[str, float]
            if key in row_index:
                paired = np.asarray(paired_matrix[row_index[key]], dtype=np.float32)[: len(window["rna_seq"])]
                struct_features = compute_structure_features(window["rna_seq"], paired)
            else:
                missing_structure_keys += 1
                struct_features = {
                    "longest_high_paired_run": math.nan,
                    "high_paired_run_count": math.nan,
                    "paired_run_density": math.nan,
                    "local_41nt_paired_mean": math.nan,
                    "local_81nt_paired_mean": math.nan,
                    "AU_rich_paired_density": math.nan,
                    "U_rich_paired_density": math.nan,
                }
            per_window_extra[key] = {**seq_features, **struct_features}

    rows: List[Dict[str, object]] = []
    for gene_row in gene_table.itertuples(index=False):
        gene_id = str(gene_row.gene_id)
        windows = top_windows.get(gene_id, [])
        top3 = windows[:3]
        top5 = windows[:5]
        agg = agg_map.get(gene_id, {})

        def collect(window_subset: Sequence[Dict[str, object]], field: str) -> List[float]:
            values: List[float] = []
            for window in window_subset:
                if field in window:
                    values.append(float(window[field]))
                else:
                    key = (window["transcript_id"], window["window_start"], window["window_end"])
                    values.append(float(per_window_extra.get(key, {}).get(field, math.nan)))
            return values

        row = {
            "gene_id": gene_id,
            "transcript_id": str(gene_row.transcript_id),
            "best_window_start": int(gene_row.best_window_start),
            "best_window_end": int(gene_row.best_window_end),
            "best_window_seq": str(gene_row.best_window_seq),
            "max_score": float(gene_row.max_score),
            "top3_mean_score": float(gene_row.top3_mean_score),
            "top5_mean_score": float(gene_row.top5_mean_score),
            "top10_mean_score": float(gene_row.top10_mean_score),
            "top5_percent_mean_score": float(gene_row.top5_percent_mean_score),
            "n_windows": int(gene_row.n_windows),
            "max_base_score": float(agg.get("max_base_score", math.nan)),
            "min_base_score": float(agg.get("min_base_score", math.nan)),
            "mean_top3_inverted_base_score": nanmean_list(collect(top3, "inverted_base_score")),
            "mean_top5_inverted_base_score": nanmean_list(collect(top5, "inverted_base_score")),
            "top3_motif_score": nanmean_list(collect(top3, "motif_match_score")),
            "top5_motif_score": nanmean_list(collect(top5, "motif_match_score")),
            "max_motif_score": float(agg.get("max_motif_score", math.nan)),
            "motif_positive_fraction_top3_mean": nanmean_list(collect(top3, "motif_positive_fraction")),
            "top3_structure_score": nanmean_list(collect(top3, "structure_match_raw")),
            "top5_structure_score": nanmean_list(collect(top5, "structure_match_raw")),
            "max_structure_score": float(agg.get("max_structure_score", math.nan)),
            "top3_paired_probability_mean": nanmean_list(collect(top3, "paired_probability_mean")),
            "top3_fraction_high_paired": nanmean_list(collect(top3, "fraction_high_paired")),
            "local_41nt_paired_mean_top3": nanmean_list(collect(top3, "local_41nt_paired_mean")),
            "local_81nt_paired_mean_top3": nanmean_list(collect(top3, "local_81nt_paired_mean")),
            "longest_high_paired_run_max": nanmax_list(collect(top3, "longest_high_paired_run")),
            "paired_run_density_top3": nanmean_list(collect(top3, "paired_run_density")),
            "reverse_complement_kmer_pair_density_top3": nanmean_list(collect(top3, "reverse_complement_kmer_pair_density")),
            "AU_rich_paired_density_top3": nanmean_list(collect(top3, "AU_rich_paired_density")),
            "U_rich_paired_density_top3": nanmean_list(collect(top3, "U_rich_paired_density")),
            "U_content_top3": nanmean_list(collect(top3, "U_content")),
            "UC_content_top3": nanmean_list(collect(top3, "UC_content")),
            "AU_content_top3": nanmean_list(collect(top3, "AU_content")),
            "max_U_homopolymer_len_max": nanmax_list(collect(top3, "max_homopolymer_U_len")),
            "shannon_entropy_top3": nanmean_list(collect(top3, "shannon_entropy")),
            "low_complexity_score_top3": nanmean_list(collect(top3, "low_complexity_score")),
            "UUUUUUU_count_max": nanmax_list(collect(top3, "UUUUUUU_count")),
            "top_Urich_kmer_density_top3": nanmean_list(collect(top3, "top_Urich_kmer_density")),
            "log1p_n_windows": float(np.log1p(int(gene_row.n_windows))),
            "n_high_score_windows_0.9": int(agg.get("n_high_score_windows_0.9", 0)),
            "n_high_score_windows_0.95": int(agg.get("n_high_score_windows_0.95", 0)),
        }
        row["fraction_high_score_windows_0.9"] = row["n_high_score_windows_0.9"] / row["n_windows"] if row["n_windows"] > 0 else math.nan
        row["fraction_high_score_windows_0.95"] = row["n_high_score_windows_0.95"] / row["n_windows"] if row["n_windows"] > 0 else math.nan
        rows.append(row)

    feature_table = pd.DataFrame(rows)
    feature_table["n_windows_bin"] = pd.qcut(feature_table["n_windows"], q=20, duplicates="drop")
    feature_table["n_windows_bin"] = feature_table["n_windows_bin"].astype(str)
    feature_table["n_windows_bin_code"] = pd.Categorical(feature_table["n_windows_bin"]).codes.astype(float)
    feature_table.to_csv(out_path, sep="\t", index=False)

    meta = {
        "n_genes": int(len(feature_table)),
        "window_chunksize": int(chunksize),
        "target_top3_windows": int(len(target_keys)),
        "matched_structure_keys": int(len(row_index)),
        "missing_structure_keys": int(missing_structure_keys),
        "duplicate_structure_keys": int(duplicate_count),
        "selected_urich_kmers": urich_kmers,
    }
    return feature_table, meta


def run_multiobjective_calibration(
    score_dir: Path,
    truth_gene_ids: set[str],
    out_dir: Path,
    n_seeds: int,
    seed0: int,
    calibration_fraction: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    score_tables: Dict[str, pd.DataFrame] = {}
    for score_name in CANDIDATE_SCORES:
        frame = pd.read_csv(score_dir / f"gene_scores_by_{score_name}.tsv", sep="\t")
        frame["gene_id"] = frame["gene_id"].astype(str)
        score_tables[score_name] = frame

    universe = set(score_tables[CANDIDATE_SCORES[0]]["gene_id"].tolist())
    comparable_truth = sorted(universe & truth_gene_ids)
    objectives = [
        "top200_overlap",
        "top200_hypergeom",
        "top200_fold",
        "weighted_top200_500_1000",
        "top500_hypergeom",
        "top1000_hypergeom",
    ]

    rng_master = np.random.default_rng(int(seed0))
    all_candidate_rows = []
    selected_rows = []
    heldout_rows = []
    raw_baseline_rows = []
    for _ in range(int(n_seeds)):
        seed = int(rng_master.integers(0, 2**31 - 1))
        rng = np.random.default_rng(seed)
        shuffled = np.asarray(comparable_truth, dtype=object).copy()
        rng.shuffle(shuffled)
        n_cal = max(1, int(round(len(shuffled) * float(calibration_fraction))))
        n_cal = min(n_cal, len(shuffled) - 1)
        calibration_truth = set(shuffled[:n_cal].tolist())
        heldout_truth = set(shuffled[n_cal:].tolist())

        split_rows = []
        for score_name, frame in score_tables.items():
            for aggregation in CANDIDATE_AGGREGATIONS:
                cal = evaluate_ranked_frame(frame, aggregation, calibration_truth, "calibration", seed, f"{score_name}__{aggregation}", TOP_KS)
                cal["score_name"] = score_name
                cal["aggregation"] = aggregation
                held = evaluate_ranked_frame(frame, aggregation, heldout_truth, "heldout", seed, f"{score_name}__{aggregation}", TOP_KS)
                held["score_name"] = score_name
                held["aggregation"] = aggregation
                split_rows.extend([cal, held])
        split_df = pd.concat(split_rows, ignore_index=True)
        all_candidate_rows.append(split_df)

        baseline = split_df[
            (split_df["split_name"] == "heldout")
            & (split_df["score_name"] == "inverted_base_motif_structure_score")
            & (split_df["aggregation"] == "top3_mean_score")
        ].copy()
        baseline["objective"] = "fixed_raw_baseline"
        raw_baseline_rows.append(baseline)

        calibration_rows = split_df[split_df["split_name"] == "calibration"].copy()
        heldout_frame = split_df[split_df["split_name"] == "heldout"].copy()
        for objective in objectives:
            best = select_candidate_by_objective(calibration_rows, objective)
            selected_rows.append(
                {
                    "seed": seed,
                    "objective": objective,
                    "selected_score_name": best["score_name"],
                    "selected_aggregation": best["aggregation"],
                    "top200_overlap": best["top200_overlap"],
                    "top200_fold": best["top200_fold"],
                    "top200_hypergeom": best["top200_hypergeom"],
                    "top500_overlap": best["top500_overlap"],
                    "top500_hypergeom": best["top500_hypergeom"],
                    "top1000_overlap": best["top1000_overlap"],
                    "top1000_hypergeom": best["top1000_hypergeom"],
                    "weighted_top200_500_1000": best["weighted_top200_500_1000"],
                    "n_calibration_truth": int(len(calibration_truth)),
                    "n_heldout_truth": int(len(heldout_truth)),
                }
            )
            held = heldout_frame[
                (heldout_frame["score_name"] == best["score_name"])
                & (heldout_frame["aggregation"] == best["aggregation"])
            ].copy()
            held["objective"] = objective
            heldout_rows.append(held)

    all_candidates = pd.concat(all_candidate_rows, ignore_index=True)
    all_candidates.to_csv(out_dir / "multiobjective_all_candidates.tsv", sep="\t", index=False)

    selected_df = pd.DataFrame(selected_rows)
    selected_df.to_csv(out_dir / "selected_combos_by_objective.tsv", sep="\t", index=False)

    heldout_df = pd.concat(heldout_rows, ignore_index=True)
    heldout_df.to_csv(out_dir / "heldout_by_objective_summary.tsv", sep="\t", index=False)

    raw_baseline_df = pd.concat(raw_baseline_rows, ignore_index=True)
    raw_baseline_df.to_csv(out_dir / "raw_baseline_heldout_summary.tsv", sep="\t", index=False)

    objective_comparison = (
        heldout_df.groupby(["objective", "top_k"], sort=False)
        .agg(
            n_seeds=("seed", "nunique"),
            mean_overlap=("overlap", "mean"),
            median_overlap=("overlap", "median"),
            min_overlap=("overlap", "min"),
            max_overlap=("overlap", "max"),
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
            mean_fold_enrichment=("fold_enrichment", "mean"),
            median_hypergeom_pvalue=("hypergeom_pvalue", "median"),
        )
        .reset_index()
    )
    objective_comparison.to_csv(out_dir / "objective_comparison.tsv", sep="\t", index=False)
    return all_candidates, heldout_df, raw_baseline_df


def sample_window_matched_negatives(
    feature_table: pd.DataFrame,
    candidate_gene_ids: set[str],
    positive_gene_ids: Sequence[str],
    rng: np.random.Generator,
) -> List[str]:
    by_bin: Dict[str, List[str]] = defaultdict(list)
    candidate_frame = feature_table[feature_table["gene_id"].isin(candidate_gene_ids)][["gene_id", "n_windows_bin_code"]].copy()
    candidate_frame["bin_key"] = candidate_frame["n_windows_bin_code"].fillna(-1).astype(int)
    for row in candidate_frame.itertuples(index=False):
        by_bin[int(row.bin_key)].append(str(row.gene_id))
    for key in by_bin:
        rng.shuffle(by_bin[key])

    selected: List[str] = []
    used: set[str] = set()
    pos_bins = feature_table.set_index("gene_id").loc[list(positive_gene_ids), "n_windows_bin_code"].fillna(-1).astype(int).tolist()
    available_bins = sorted(by_bin)
    for pos_bin in pos_bins:
        candidate = None
        search_order = sorted(available_bins, key=lambda x: abs(x - int(pos_bin)))
        for bin_key in search_order:
            while by_bin.get(bin_key):
                gene_id = by_bin[bin_key].pop()
                if gene_id not in used:
                    candidate = gene_id
                    break
            if candidate is not None:
                break
        if candidate is not None:
            selected.append(candidate)
            used.add(candidate)
    return selected


def build_model_pipeline(model_name: str) -> Pipeline:
    if model_name == "logistic_l2":
        classifier = LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            class_weight="balanced",
            max_iter=4000,
            random_state=0,
        )
    elif model_name == "logistic_elasticnet":
        classifier = LogisticRegression(
            penalty="elasticnet",
            C=1.0,
            solver="saga",
            l1_ratio=0.5,
            class_weight="balanced",
            max_iter=8000,
            random_state=0,
        )
    else:
        raise ValueError(model_name)

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", classifier),
        ]
    )


def run_supervised_calibration(
    feature_table: pd.DataFrame,
    truth_gene_ids: set[str],
    out_dir: Path,
    n_seeds: int,
    seed0: int,
    calibration_fraction: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    feature_table = feature_table.copy()
    feature_table["gene_id"] = feature_table["gene_id"].astype(str)
    comparable_truth = sorted(set(feature_table["gene_id"]) & truth_gene_ids)
    full_truth = set(comparable_truth)

    raw_rank = feature_table.sort_values(["top3_mean_score", "gene_id"], ascending=[False, True]).reset_index(drop=True)
    hard_negative_pool_all = set(raw_rank.head(1000).loc[~raw_rank["gene_id"].isin(full_truth), "gene_id"].tolist())

    feature_indexed = feature_table.set_index("gene_id", drop=False)
    X_all = feature_indexed[FEATURE_COLUMNS]

    rng_master = np.random.default_rng(int(seed0))
    metric_rows = []
    coef_rows = []
    pred_rows = []
    model_feature_importance_rows = []
    available_models = ["logistic_l2", "logistic_elasticnet"]
    for _ in range(int(n_seeds)):
        seed = int(rng_master.integers(0, 2**31 - 1))
        rng = np.random.default_rng(seed)
        shuffled = np.asarray(comparable_truth, dtype=object).copy()
        rng.shuffle(shuffled)
        n_cal = max(1, int(round(len(shuffled) * float(calibration_fraction))))
        n_cal = min(n_cal, len(shuffled) - 1)
        calibration_truth = set(shuffled[:n_cal].tolist())
        heldout_truth = set(shuffled[n_cal:].tolist())

        train_exclude = heldout_truth
        available_gene_ids = set(feature_table["gene_id"]) - calibration_truth - train_exclude
        hard_negatives = list((hard_negative_pool_all & available_gene_ids))
        rng.shuffle(hard_negatives)
        hard_negatives = hard_negatives[: min(len(calibration_truth), len(hard_negatives))]

        matched_candidates = available_gene_ids - set(hard_negatives)
        matched_negatives = sample_window_matched_negatives(
            feature_table=feature_table,
            candidate_gene_ids=matched_candidates,
            positive_gene_ids=sorted(calibration_truth),
            rng=rng,
        )

        random_candidates = list(available_gene_ids - set(hard_negatives) - set(matched_negatives))
        rng.shuffle(random_candidates)
        random_negatives = random_candidates[: min(len(calibration_truth), len(random_candidates))]

        train_negative_ids = sorted(set(hard_negatives) | set(matched_negatives) | set(random_negatives))
        train_gene_ids = sorted(calibration_truth) + train_negative_ids
        y_train = np.asarray([1] * len(calibration_truth) + [0] * len(train_negative_ids), dtype=np.int64)
        X_train = X_all.loc[train_gene_ids]

        role_map = {}
        for gene_id in feature_table["gene_id"]:
            if gene_id in calibration_truth:
                role_map[gene_id] = "calibration_truth"
            elif gene_id in heldout_truth:
                role_map[gene_id] = "heldout_truth"
            elif gene_id in train_negative_ids:
                role_map[gene_id] = "train_negative"
            else:
                role_map[gene_id] = "unlabeled"

        for model_name in available_models:
            pipe = build_model_pipeline(model_name)
            pipe.fit(X_train, y_train)
            scores = pipe.predict_proba(X_all)[:, 1]
            pred_frame = feature_table[["gene_id"]].copy()
            pred_frame["prediction_score"] = scores

            cal_metrics = evaluate_ranked_frame(pred_frame, "prediction_score", calibration_truth, "calibration", seed, model_name, TOP_KS)
            cal_metrics["model_name"] = model_name
            held_metrics = evaluate_ranked_frame(pred_frame, "prediction_score", heldout_truth, "heldout", seed, model_name, TOP_KS)
            held_metrics["model_name"] = model_name
            metric_rows.extend([cal_metrics, held_metrics])

            model = pipe.named_steps["model"]
            coef = model.coef_[0]
            abs_coef = np.abs(coef)
            for feature_name, value, abs_value in zip(FEATURE_COLUMNS, coef.tolist(), abs_coef.tolist()):
                coef_rows.append(
                    {
                        "seed": seed,
                        "model_name": model_name,
                        "feature_name": feature_name,
                        "coefficient": float(value),
                        "abs_coefficient": float(abs_value),
                    }
                )
            model_feature_importance_rows.extend(
                {
                    "seed": seed,
                    "model_name": model_name,
                    "feature_name": feature_name,
                    "importance": float(abs_value),
                }
                for feature_name, abs_value in zip(FEATURE_COLUMNS, abs_coef.tolist())
            )

            pred_tmp = pred_frame.copy()
            pred_tmp["seed"] = seed
            pred_tmp["model_name"] = model_name
            pred_tmp["role"] = pred_tmp["gene_id"].map(role_map)
            pred_rows.append(pred_tmp)

    metric_df = pd.concat(metric_rows, ignore_index=True)
    metric_df.to_csv(out_dir / "calibration_model_heldout_summary.tsv", sep="\t", index=False)

    coef_df = pd.DataFrame(coef_rows)
    coef_df.to_csv(out_dir / "calibration_model_coefficients.tsv", sep="\t", index=False)

    importance_df = (
        pd.DataFrame(model_feature_importance_rows)
        .groupby(["model_name", "feature_name"], sort=False)
        .agg(mean_importance=("importance", "mean"), median_importance=("importance", "median"))
        .reset_index()
        .sort_values(["model_name", "mean_importance"], ascending=[True, False])
    )
    importance_df.to_csv(out_dir / "calibration_model_feature_importance.tsv", sep="\t", index=False)

    predictions_df = pd.concat(pred_rows, ignore_index=True)
    predictions_df.to_csv(out_dir / "calibration_model_predictions.tsv", sep="\t", index=False)
    return metric_df, importance_df


def run_two_branch_validation(
    feature_table: pd.DataFrame,
    truth_gene_ids: set[str],
    out_dir: Path,
    n_seeds: int,
    seed0: int,
    calibration_fraction: float,
) -> pd.DataFrame:
    frame = feature_table[["gene_id", "top3_mean_score"] + B_BRANCH_FEATURES].copy()
    frame["A_branch_score"] = percentile_rank(frame["top3_mean_score"].astype(float))
    b_percentiles = []
    for feature_name in B_BRANCH_FEATURES:
        b_percentiles.append(percentile_rank(frame[feature_name].astype(float)))
    frame["B_branch_score"] = pd.concat(b_percentiles, axis=1).mean(axis=1, skipna=True)
    frame["final_score_max"] = frame[["A_branch_score", "B_branch_score"]].max(axis=1)
    frame["final_score_0.7A_0.3B"] = 0.7 * frame["A_branch_score"] + 0.3 * frame["B_branch_score"]
    frame["final_score_0.5A_0.5B"] = 0.5 * frame["A_branch_score"] + 0.5 * frame["B_branch_score"]
    frame["final_score_0.3A_0.7B"] = 0.3 * frame["A_branch_score"] + 0.7 * frame["B_branch_score"]

    comparable_truth = sorted(set(frame["gene_id"]) & truth_gene_ids)
    rng_master = np.random.default_rng(int(seed0))
    rows = []
    score_cols = [
        "A_branch_score",
        "B_branch_score",
        "final_score_max",
        "final_score_0.7A_0.3B",
        "final_score_0.5A_0.5B",
        "final_score_0.3A_0.7B",
    ]
    for _ in range(int(n_seeds)):
        seed = int(rng_master.integers(0, 2**31 - 1))
        rng = np.random.default_rng(seed)
        shuffled = np.asarray(comparable_truth, dtype=object).copy()
        rng.shuffle(shuffled)
        n_cal = max(1, int(round(len(shuffled) * float(calibration_fraction))))
        n_cal = min(n_cal, len(shuffled) - 1)
        heldout_truth = set(shuffled[n_cal:].tolist())
        for score_col in score_cols:
            result = evaluate_ranked_frame(frame[["gene_id", score_col]], score_col, heldout_truth, "heldout", seed, score_col, TOP_KS)
            result["score_variant"] = score_col
            rows.append(result)
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(out_dir / "two_branch_heldout_summary.tsv", sep="\t", index=False)
    return out


def render_review_summary(
    out_dir: Path,
    objective_comparison: pd.DataFrame,
    raw_baseline_heldout: pd.DataFrame,
    supervised_metrics: pd.DataFrame,
    two_branch_metrics: pd.DataFrame,
) -> None:
    raw_top = raw_baseline_heldout.groupby("top_k").agg(
        mean_overlap=("overlap", "mean"),
        mean_precision=("precision", "mean"),
        mean_fold_enrichment=("fold_enrichment", "mean"),
    )

    objective_top200 = objective_comparison[objective_comparison["top_k"] == 200].sort_values(
        ["mean_overlap", "mean_fold_enrichment", "median_hypergeom_pvalue"],
        ascending=[False, False, True],
    )
    supervised_heldout = (
        supervised_metrics[supervised_metrics["split_name"] == "heldout"]
        .groupby(["model_name", "top_k"], sort=False)
        .agg(
            mean_overlap=("overlap", "mean"),
            median_overlap=("overlap", "median"),
            min_overlap=("overlap", "min"),
            max_overlap=("overlap", "max"),
            mean_precision=("precision", "mean"),
            mean_fold_enrichment=("fold_enrichment", "mean"),
            median_hypergeom_pvalue=("hypergeom_pvalue", "median"),
        )
        .reset_index()
    )
    supervised_top200 = supervised_heldout[supervised_heldout["top_k"] == 200].sort_values(
        ["mean_overlap", "mean_fold_enrichment", "median_hypergeom_pvalue"],
        ascending=[False, False, True],
    )
    two_branch_agg = (
        two_branch_metrics.groupby(["score_variant", "top_k"], sort=False)
        .agg(
            mean_overlap=("overlap", "mean"),
            median_overlap=("overlap", "median"),
            min_overlap=("overlap", "min"),
            max_overlap=("overlap", "max"),
            mean_precision=("precision", "mean"),
            mean_fold_enrichment=("fold_enrichment", "mean"),
            median_hypergeom_pvalue=("hypergeom_pvalue", "median"),
        )
        .reset_index()
    )
    two_branch_top200 = two_branch_agg[two_branch_agg["top_k"] == 200].sort_values(
        ["mean_overlap", "mean_fold_enrichment", "median_hypergeom_pvalue"],
        ascending=[False, False, True],
    )

    best_supervised = supervised_top200.iloc[0] if not supervised_top200.empty else None
    best_two_branch = two_branch_top200.iloc[0] if not two_branch_top200.empty else None
    baseline_top200 = raw_baseline_heldout[raw_baseline_heldout["top_k"] == 200]
    baseline_interval = (
        int(baseline_top200["overlap"].min()),
        float(baseline_top200["overlap"].median()),
        int(baseline_top200["overlap"].max()),
    )

    lines = [
        "# REVIEW V4 SUMMARY",
        "",
        "## Raw Baseline",
        "",
        "Oracle/full-truth diagnostic from V2:",
        "- Top200 = 16",
        "- Top500 = 28",
        "- Top1000 = 48",
        "",
        "Held-out fixed raw baseline (`inverted_base_motif_structure_score + top3_mean_score`):",
    ]
    for top_k in [200, 500, 1000]:
        row = raw_top.loc[top_k]
        lines.append(
            f"- Top{top_k}: mean overlap = {row['mean_overlap']:.2f}, mean precision = {row['mean_precision']:.4f}, mean fold_enrichment = {row['mean_fold_enrichment']:.2f}"
        )

    lines.extend(
        [
            "",
            "## Multi-objective Calibration",
            "",
            "Held-out Top200 comparison by objective:",
        ]
    )
    for row in objective_top200.itertuples(index=False):
        lines.append(
            f"- {row.objective}: mean overlap = {row.mean_overlap:.2f}, precision = {row.mean_precision:.4f}, fold = {row.mean_fold_enrichment:.2f}, median p = {row.median_hypergeom_pvalue:.3e}"
        )

    lines.extend(["", "## Supervised Calibration", ""])
    if best_supervised is not None:
        lines.append(
            f"Best held-out Top200 model: {best_supervised.model_name}, mean overlap = {best_supervised.mean_overlap:.2f}, interval = [{int(best_supervised.min_overlap)}, {best_supervised.median_overlap:.1f}, {int(best_supervised.max_overlap)}], mean fold = {best_supervised.mean_fold_enrichment:.2f}"
        )
    lines.append("- LightGBM: unavailable in current environment, not run.")
    if best_supervised is not None:
        baseline_mean_top200 = float(raw_top.loc[200, "mean_overlap"])
        lines.append(
            f"- Logistic models {'exceed' if best_supervised.mean_overlap > baseline_mean_top200 else 'do not exceed'} the held-out raw baseline Top200 mean ({baseline_mean_top200:.2f})."
        )

    lines.extend(["", "## Two-branch Score", ""])
    if best_two_branch is not None:
        lines.append(
            f"Best held-out Top200 variant: {best_two_branch.score_variant}, mean overlap = {best_two_branch.mean_overlap:.2f}, interval = [{int(best_two_branch.min_overlap)}, {best_two_branch.median_overlap:.1f}, {int(best_two_branch.max_overlap)}], mean fold = {best_two_branch.mean_fold_enrichment:.2f}"
        )
        baseline_mean_top200 = float(raw_top.loc[200, "mean_overlap"])
        lines.append(
            f"- Two-branch {'exceeds' if best_two_branch.mean_overlap > baseline_mean_top200 else 'does not exceed'} the held-out raw baseline Top200 mean ({baseline_mean_top200:.2f})."
        )

    lines.extend(
        [
            "",
            "## Current Ceiling",
            "",
            f"- Held-out raw baseline Top200 stable interval: [{baseline_interval[0]}, {baseline_interval[1]:.1f}, {baseline_interval[2]}]",
        ]
    )
    if best_supervised is not None and best_two_branch is not None:
        stable_best = max(float(best_supervised.median_overlap), float(best_two_branch.median_overlap), baseline_interval[1])
        lines.append(f"- Existing features currently stabilize around Top200 overlap {stable_best:.1f} on held-out splits.")
    lines.extend(
        [
            "- Main conclusions should still be based on held-out validation, not oracle/full-truth diagnostics.",
            "- If V4 still cannot materially push held-out Top200 above the raw baseline, region/expression modules are still justified.",
            "",
        ]
    )
    (out_dir / "REVIEW_V4_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gene-score-dir", required=True)
    parser.add_argument("--truth-gene-list", required=True)
    parser.add_argument("--window-scores", required=True)
    parser.add_argument("--window-table", required=True)
    parser.add_argument("--structure-npy", required=True)
    parser.add_argument("--motif-top-kmers", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--seed0", type=int, default=20260605)
    parser.add_argument("--calibration-fraction", type=float, default=0.7)
    parser.add_argument("--window-chunksize", type=int, default=200000)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    truth_gene_ids = load_truth_gene_ids(Path(args.truth_gene_list))
    score_dir = Path(args.gene_score_dir)

    all_candidates, objective_heldout, raw_baseline = run_multiobjective_calibration(
        score_dir=score_dir,
        truth_gene_ids=truth_gene_ids,
        out_dir=out_dir,
        n_seeds=int(args.n_seeds),
        seed0=int(args.seed0),
        calibration_fraction=float(args.calibration_fraction),
    )

    feature_table, feature_meta = build_existing_feature_gene_table(
        gene_table_path=score_dir / "gene_scores_by_inverted_base_motif_structure_score.tsv",
        window_scores_path=Path(args.window_scores),
        window_table_path=Path(args.window_table),
        structure_npy_path=Path(args.structure_npy),
        motif_top_kmers_path=Path(args.motif_top_kmers),
        out_path=out_dir / "existing_feature_gene_table.tsv",
        chunksize=int(args.window_chunksize),
    )
    (out_dir / "existing_feature_gene_table_meta.json").write_text(json.dumps(feature_meta, indent=2, ensure_ascii=False), encoding="utf-8")

    supervised_metrics, importance_df = run_supervised_calibration(
        feature_table=feature_table,
        truth_gene_ids=truth_gene_ids,
        out_dir=out_dir,
        n_seeds=int(args.n_seeds),
        seed0=int(args.seed0),
        calibration_fraction=float(args.calibration_fraction),
    )

    two_branch_metrics = run_two_branch_validation(
        feature_table=feature_table,
        truth_gene_ids=truth_gene_ids,
        out_dir=out_dir,
        n_seeds=int(args.n_seeds),
        seed0=int(args.seed0),
        calibration_fraction=float(args.calibration_fraction),
    )

    objective_comparison = pd.read_csv(out_dir / "objective_comparison.tsv", sep="\t")
    render_review_summary(
        out_dir=out_dir,
        objective_comparison=objective_comparison,
        raw_baseline_heldout=raw_baseline,
        supervised_metrics=supervised_metrics,
        two_branch_metrics=two_branch_metrics,
    )

    meta = {
        "gene_score_dir": str(score_dir),
        "truth_gene_list": str(args.truth_gene_list),
        "window_scores": str(args.window_scores),
        "window_table": str(args.window_table),
        "structure_npy": str(args.structure_npy),
        "motif_top_kmers": str(args.motif_top_kmers),
        "n_seeds": int(args.n_seeds),
        "seed0": int(args.seed0),
        "calibration_fraction": float(args.calibration_fraction),
        "candidate_scores": CANDIDATE_SCORES,
        "candidate_aggregations": CANDIDATE_AGGREGATIONS,
        "feature_columns": FEATURE_COLUMNS,
        "b_branch_features": B_BRANCH_FEATURES,
        "weighted_objective_definition": "0.5*-log10(top200_hypergeom)+0.3*-log10(top500_hypergeom)+0.2*-log10(top1000_hypergeom)",
        "lightgbm_status": "not_available",
    }
    (out_dir / "review_v4_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
