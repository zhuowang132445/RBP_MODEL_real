#!/usr/bin/env python3
"""OsDRB1-specific calibration/held-out validation without retraining."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

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


def evaluate_gene_table(
    frame: pd.DataFrame,
    truth_gene_ids: set[str],
    split_name: str,
    seed: int,
    score_name: str,
    aggregation: str,
    top_ks: Iterable[int],
) -> pd.DataFrame:
    ranked = frame.sort_values([aggregation, "gene_id"], ascending=[False, True], na_position="last").reset_index(drop=True)
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
                "split_name": split_name,
                "score_name": score_name,
                "aggregation": aggregation,
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


def select_best_candidate(calibration_rows: pd.DataFrame) -> pd.Series:
    top1000 = calibration_rows[calibration_rows["top_k"] == 1000].copy()
    top500 = calibration_rows[calibration_rows["top_k"] == 500][["score_name", "aggregation", "hypergeom_pvalue", "overlap"]].rename(
        columns={
            "hypergeom_pvalue": "top500_hypergeom_pvalue",
            "overlap": "top500_overlap",
        }
    )
    merged = top1000.merge(top500, on=["score_name", "aggregation"], how="left")
    merged["selection_metric"] = -np.log10(np.clip(merged["hypergeom_pvalue"].to_numpy(dtype=float), 1e-300, 1.0))
    merged = merged.sort_values(
        [
            "selection_metric",
            "overlap",
            "fold_enrichment",
            "top500_overlap",
            "top500_hypergeom_pvalue",
        ],
        ascending=[False, False, False, False, True],
        na_position="last",
    ).reset_index(drop=True)
    return merged.iloc[0]


def add_bin_adjusted_score(frame: pd.DataFrame, score_col: str, n_bins: int) -> pd.DataFrame:
    out = frame.copy()
    out = out.sort_values(["n_windows", "gene_id"], ascending=[True, True]).reset_index(drop=True)
    out["window_count_bin"] = pd.qcut(out["n_windows"], q=n_bins, duplicates="drop")
    grouped = out.groupby("window_count_bin", observed=True)
    out["bin_adjusted_top3_mean_score"] = grouped[score_col].rank(method="average", pct=True)
    bin_mean = grouped[score_col].transform("mean")
    bin_std = grouped[score_col].transform("std").replace(0.0, np.nan)
    out["bin_adjusted_top3_mean_zscore"] = (out[score_col] - bin_mean) / bin_std
    return out


def summarize_correlation(frame: pd.DataFrame, score_col: str) -> Dict[str, float]:
    pair = frame[["n_windows", score_col]].dropna()
    if len(pair) < 2:
        return {
            "pearson_n_windows_vs_score": float("nan"),
            "spearman_n_windows_vs_score": float("nan"),
        }
    return {
        "pearson_n_windows_vs_score": float(pair["n_windows"].corr(pair[score_col], method="pearson")),
        "spearman_n_windows_vs_score": float(pair["n_windows"].corr(pair[score_col], method="spearman")),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gene-score-dir", required=True)
    parser.add_argument("--truth-gene-list", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--seed0", type=int, default=20260605)
    parser.add_argument("--calibration-fraction", type=float, default=0.7)
    parser.add_argument("--n-window-bins", type=int, default=20)
    parser.add_argument("--bin-adjust-score-name", default="inverted_base_motif_structure_score")
    parser.add_argument("--bin-adjust-aggregation", default="top3_mean_score")
    args = parser.parse_args()

    score_dir = Path(args.gene_score_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    truth_gene_ids = load_truth_gene_ids(Path(args.truth_gene_list))
    score_tables: Dict[str, pd.DataFrame] = {}
    for score_name in CANDIDATE_SCORES:
        path = score_dir / f"gene_scores_by_{score_name}.tsv"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path, sep="\t")
        frame["gene_id"] = frame["gene_id"].astype(str)
        score_tables[score_name] = frame

    universe = set(score_tables[CANDIDATE_SCORES[0]]["gene_id"].tolist())
    comparable_truth = sorted(universe & truth_gene_ids)

    all_rows = []
    selected_rows = []
    selected_combo_rows = []
    rng_master = np.random.default_rng(int(args.seed0))
    for split_idx in range(int(args.n_seeds)):
        seed = int(rng_master.integers(0, 2**31 - 1))
        rng = np.random.default_rng(seed)
        shuffled = np.asarray(comparable_truth, dtype=object).copy()
        rng.shuffle(shuffled)
        n_cal = max(1, int(round(len(shuffled) * float(args.calibration_fraction))))
        n_cal = min(n_cal, len(shuffled) - 1)
        calibration_truth = set(shuffled[:n_cal].tolist())
        heldout_truth = set(shuffled[n_cal:].tolist())

        split_rows = []
        for score_name, frame in score_tables.items():
            for aggregation in CANDIDATE_AGGREGATIONS:
                split_rows.append(
                    evaluate_gene_table(
                        frame=frame,
                        truth_gene_ids=calibration_truth,
                        split_name="calibration",
                        seed=seed,
                        score_name=score_name,
                        aggregation=aggregation,
                        top_ks=TOP_KS,
                    )
                )
                split_rows.append(
                    evaluate_gene_table(
                        frame=frame,
                        truth_gene_ids=heldout_truth,
                        split_name="heldout",
                        seed=seed,
                        score_name=score_name,
                        aggregation=aggregation,
                        top_ks=TOP_KS,
                    )
                )
        split_df = pd.concat(split_rows, ignore_index=True)
        all_rows.append(split_df)

        best = select_best_candidate(split_df[split_df["split_name"] == "calibration"])
        selected_combo_rows.append(
            {
                "seed": seed,
                "selected_score_name": best["score_name"],
                "selected_aggregation": best["aggregation"],
                "calibration_top1000_overlap": int(best["overlap"]),
                "calibration_top1000_fold_enrichment": float(best["fold_enrichment"]),
                "calibration_top1000_hypergeom_pvalue": float(best["hypergeom_pvalue"]),
                "calibration_top500_overlap": int(best["top500_overlap"]),
                "calibration_top500_hypergeom_pvalue": float(best["top500_hypergeom_pvalue"]),
                "n_calibration_truth": int(len(calibration_truth)),
                "n_heldout_truth": int(len(heldout_truth)),
            }
        )
        heldout_best = split_df[
            (split_df["split_name"] == "heldout")
            & (split_df["score_name"] == best["score_name"])
            & (split_df["aggregation"] == best["aggregation"])
        ].copy()
        heldout_best["selected_by_calibration"] = True
        selected_rows.append(heldout_best)

    all_results = pd.concat(all_rows, ignore_index=True)
    all_results.to_csv(out_dir / "heldout_validation_all_candidates.tsv", sep="\t", index=False)

    heldout_summary = pd.concat(selected_rows, ignore_index=True)
    heldout_summary.to_csv(out_dir / "heldout_validation_summary.tsv", sep="\t", index=False)
    pd.DataFrame(selected_combo_rows).to_csv(out_dir / "heldout_validation_selected_combos.tsv", sep="\t", index=False)

    selected_combo_summary = (
        heldout_summary.groupby(["score_name", "aggregation", "top_k"], sort=False)
        .agg(
            n_selected=("seed", "nunique"),
            mean_overlap=("overlap", "mean"),
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
            mean_fold_enrichment=("fold_enrichment", "mean"),
            median_hypergeom_pvalue=("hypergeom_pvalue", "median"),
        )
        .reset_index()
    )
    selected_combo_summary.to_csv(out_dir / "heldout_validation_selected_combo_summary.tsv", sep="\t", index=False)

    # Window-count bin adjustment on the fixed best-review_v2 combo.
    base_frame = score_tables[str(args.bin_adjust_score_name)].copy()
    adjusted = add_bin_adjusted_score(
        frame=base_frame,
        score_col=str(args.bin_adjust_aggregation),
        n_bins=int(args.n_window_bins),
    )
    adjusted.to_csv(out_dir / "bin_adjusted_gene_scores.tsv", sep="\t", index=False)

    raw_eval = evaluate_gene_table(
        frame=adjusted.rename(columns={str(args.bin_adjust_aggregation): "raw_top3_mean_score"}),
        truth_gene_ids=set(comparable_truth),
        split_name="full_truth_raw",
        seed=-1,
        score_name=str(args.bin_adjust_score_name),
        aggregation="raw_top3_mean_score",
        top_ks=TOP_KS,
    )
    adj_eval = evaluate_gene_table(
        frame=adjusted,
        truth_gene_ids=set(comparable_truth),
        split_name="full_truth_bin_adjusted",
        seed=-1,
        score_name=str(args.bin_adjust_score_name),
        aggregation="bin_adjusted_top3_mean_score",
        top_ks=TOP_KS,
    )
    raw_corr = summarize_correlation(adjusted.rename(columns={str(args.bin_adjust_aggregation): "raw_top3_mean_score"}), "raw_top3_mean_score")
    adj_corr = summarize_correlation(adjusted, "bin_adjusted_top3_mean_score")
    raw_eval["comparison_name"] = "raw_top3_mean_score"
    adj_eval["comparison_name"] = "bin_adjusted_top3_mean_score"
    raw_eval["pearson_n_windows_vs_score"] = raw_corr["pearson_n_windows_vs_score"]
    raw_eval["spearman_n_windows_vs_score"] = raw_corr["spearman_n_windows_vs_score"]
    adj_eval["pearson_n_windows_vs_score"] = adj_corr["pearson_n_windows_vs_score"]
    adj_eval["spearman_n_windows_vs_score"] = adj_corr["spearman_n_windows_vs_score"]
    bin_summary = pd.concat([raw_eval, adj_eval], ignore_index=True)
    bin_summary.to_csv(out_dir / "window_count_bin_adjustment_summary.tsv", sep="\t", index=False)
    (
        adjusted.groupby("window_count_bin", observed=True)
        .agg(
            n_genes=("gene_id", "size"),
            min_n_windows=("n_windows", "min"),
            max_n_windows=("n_windows", "max"),
            raw_score_mean=(str(args.bin_adjust_aggregation), "mean"),
            raw_score_median=(str(args.bin_adjust_aggregation), "median"),
            adjusted_percentile_mean=("bin_adjusted_top3_mean_score", "mean"),
            adjusted_percentile_median=("bin_adjusted_top3_mean_score", "median"),
        )
        .reset_index()
        .to_csv(out_dir / "window_count_bin_summary.tsv", sep="\t", index=False)
    )

    meta = {
        "truth_gene_list": str(args.truth_gene_list),
        "comparable_truth_genes": int(len(comparable_truth)),
        "candidate_scores": CANDIDATE_SCORES,
        "candidate_aggregations": CANDIDATE_AGGREGATIONS,
        "selection_rule": "min calibration Top1000 hypergeom_pvalue; tie-break by Top1000 overlap, fold_enrichment, then Top500 overlap/pvalue",
        "n_seeds": int(args.n_seeds),
        "bin_adjust_score_name": str(args.bin_adjust_score_name),
        "bin_adjust_aggregation": str(args.bin_adjust_aggregation),
    }
    (out_dir / "heldout_validation_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
