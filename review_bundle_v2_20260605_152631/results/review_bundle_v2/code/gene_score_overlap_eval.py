#!/usr/bin/env python3
"""Build per-score gene aggregations and evaluate overlap against truth genes."""

from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

try:
    from scipy.stats import hypergeom
except Exception:  # pragma: no cover
    hypergeom = None


DEFAULT_SCORE_COLUMNS = [
    "base_score",
    "inverted_base_score",
    "motif_only_score",
    "structure_only_score",
    "motif_structure_score",
    "base_motif_score",
    "inverted_base_motif_score",
    "base_motif_structure_score",
    "inverted_base_motif_structure_score",
    "posthoc_score",
]

AGG_COLUMNS = [
    "max_score",
    "top3_mean_score",
    "top5_mean_score",
    "top10_mean_score",
    "top5_percent_mean_score",
    "window_count_adjusted_score",
]


def load_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".gz":
        return pd.read_csv(path, sep="\t", compression="gzip")
    return pd.read_csv(path, sep="\t")


def load_truth_gene_ids(path: Path) -> set[str]:
    if path.suffix == ".gz":
        frame = pd.read_csv(path, sep="\t", compression="gzip")
    else:
        frame = pd.read_csv(path, sep=None, engine="python")
    for column in ["gene_id", "target_gene_id", "truth_gene_id", "gene"]:
        if column in frame.columns:
            return {str(x) for x in frame[column].dropna().astype(str).tolist()}
    if len(frame.columns) == 1:
        return {str(x) for x in frame.iloc[:, 0].dropna().astype(str).tolist()}
    raise ValueError(f"cannot infer truth gene id column from {path}")


def safe_hypergeom_sf(k_minus_one: int, m: int, n: int, n_draws: int) -> float:
    if hypergeom is not None:
        return float(hypergeom.sf(k_minus_one, m, n, n_draws))
    max_k = min(n, n_draws)
    denom = math.comb(m, n_draws)
    total = 0.0
    for k in range(max(0, k_minus_one + 1), max_k + 1):
        total += (math.comb(n, k) * math.comb(m - n, n_draws - k)) / denom
    return float(min(max(total, 0.0), 1.0))


def finalize_gene_rows(
    frame: pd.DataFrame,
    score_col: str,
    score_name: str,
    window_count_penalty: float,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    ranked = frame.sort_values([score_col, "gene_id"], ascending=[False, True], na_position="last").reset_index(drop=True)
    for gene_id, group in ranked.groupby("gene_id", sort=False):
        g = group.sort_values(score_col, ascending=False, na_position="last")
        values = g[score_col].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if len(finite) == 0:
            agg = {c: float("nan") for c in AGG_COLUMNS}
        else:
            top3 = float(np.mean(finite[: min(3, len(finite))]))
            top5 = float(np.mean(finite[: min(5, len(finite))]))
            top10 = float(np.mean(finite[: min(10, len(finite))]))
            top5pct_n = max(1, int(math.ceil(len(finite) * 0.05)))
            top5pct = float(np.mean(finite[:top5pct_n]))
            max_score = float(finite[0])
            agg = {
                "max_score": max_score,
                "top3_mean_score": top3,
                "top5_mean_score": top5,
                "top10_mean_score": top10,
                "top5_percent_mean_score": top5pct,
                "window_count_adjusted_score": float(max_score - float(window_count_penalty) * math.log2(len(g) + 1.0)),
            }
        best = g.iloc[0]
        rows.append(
            {
                "score_name": score_name,
                "gene_id": str(gene_id),
                "transcript_id": str(best["transcript_id"]),
                "n_windows": int(len(g)),
                "best_window_start": int(best["window_start"]),
                "best_window_end": int(best["window_end"]),
                "best_window_seq": str(best["rna_seq"]),
                "best_window_score": float(best[score_col]) if pd.notna(best[score_col]) else float("nan"),
                "motif_score": float(best["motif_match_score"]) if "motif_match_score" in best.index and pd.notna(best["motif_match_score"]) else float("nan"),
                "structure_score": float(best["structure_match_score"]) if "structure_match_score" in best.index and pd.notna(best["structure_match_score"]) else float("nan"),
                "base_score": float(best["base_score"]) if "base_score" in best.index and pd.notna(best["base_score"]) else float("nan"),
                "paired_probability_mean": float(best["paired_probability_mean"]) if "paired_probability_mean" in best.index and pd.notna(best["paired_probability_mean"]) else float("nan"),
                "fraction_high_paired": float(best["fraction_high_paired"]) if "fraction_high_paired" in best.index and pd.notna(best["fraction_high_paired"]) else float("nan"),
                **agg,
            }
        )
    out = pd.DataFrame(rows)
    return out


def evaluate_overlap(
    gene_scores: pd.DataFrame,
    truth_gene_ids: set[str],
    score_name: str,
    agg_col: str,
    top_ks: Iterable[int],
) -> pd.DataFrame:
    ranked = gene_scores.sort_values([agg_col, "gene_id"], ascending=[False, True], na_position="last").reset_index(drop=True)
    universe = {str(x) for x in ranked["gene_id"].astype(str).tolist()}
    comparable_truth = universe & truth_gene_ids
    m = len(universe)
    n = len(comparable_truth)
    rows = []
    for top_k in top_ks:
        subset = ranked.head(min(int(top_k), len(ranked)))
        hits = sum(1 for gene_id in subset["gene_id"].astype(str).tolist() if gene_id in comparable_truth)
        expected = (len(subset) * n / m) if m else float("nan")
        precision = (hits / len(subset)) if len(subset) else float("nan")
        recall = (hits / n) if n else float("nan")
        fold = (hits / expected) if expected and np.isfinite(expected) and expected > 0 else float("nan")
        pvalue = safe_hypergeom_sf(hits - 1, m, n, len(subset)) if m and n and len(subset) else float("nan")
        rows.append(
            {
                "score_name": score_name,
                "aggregation": agg_col,
                "top_k": int(top_k),
                "universe_genes": int(m),
                "comparable_truth_genes": int(n),
                "overlap": int(hits),
                "precision": float(precision),
                "recall": float(recall),
                "expected_random": float(expected),
                "fold_enrichment": float(fold),
                "hypergeom_pvalue": float(pvalue),
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-scores", required=True)
    parser.add_argument("--truth-gene-list", required=True)
    parser.add_argument("--rbp-id", required=True)
    parser.add_argument("--score-columns", default=",".join(DEFAULT_SCORE_COLUMNS))
    parser.add_argument("--window-count-penalty", type=float, default=0.05)
    parser.add_argument("--top-ks", default="50,100,200,500,1000")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame = load_table(Path(args.window_scores))
    rbp_mask = None
    for column in ["scorer_rbp_id", "rbp_id"]:
        if column in frame.columns:
            rbp_mask = frame[column].astype(str) == str(args.rbp_id)
            break
    if rbp_mask is None:
        raise ValueError("window score table missing scorer_rbp_id/rbp_id column")
    frame = frame.loc[rbp_mask].copy()
    if frame.empty:
        raise ValueError(f"no rows for rbp_id={args.rbp_id}")

    truth_gene_ids = load_truth_gene_ids(Path(args.truth_gene_list))
    top_ks = [int(x) for x in str(args.top_ks).split(",") if str(x).strip()]
    score_columns = [x.strip() for x in str(args.score_columns).split(",") if x.strip()]

    overlap_rows = []
    score_reports = []
    for score_col in score_columns:
        if score_col not in frame.columns:
            continue
        gene_scores = finalize_gene_rows(
            frame=frame,
            score_col=score_col,
            score_name=score_col,
            window_count_penalty=args.window_count_penalty,
        )
        gene_path = out_dir / f"gene_scores_by_{score_col}.tsv"
        gene_scores.to_csv(gene_path, sep="\t", index=False)
        score_reports.append({"score_name": score_col, "gene_scores_tsv": str(gene_path), "n_genes": int(len(gene_scores))})
        for agg_col in AGG_COLUMNS:
            overlap_rows.append(
                evaluate_overlap(
                    gene_scores=gene_scores,
                    truth_gene_ids=truth_gene_ids,
                    score_name=score_col,
                    agg_col=agg_col,
                    top_ks=top_ks,
                )
            )

    if not overlap_rows:
        raise ValueError("no score columns were available for overlap evaluation")
    overlap = pd.concat(overlap_rows, ignore_index=True)
    overlap.to_csv(out_dir / "overlap_evaluation_summary.tsv", sep="\t", index=False)

    best = overlap.sort_values(
        ["top_k", "overlap", "fold_enrichment", "hypergeom_pvalue"],
        ascending=[False, False, False, True],
        na_position="last",
    ).iloc[0].to_dict()
    summary = {
        "rbp_id": str(args.rbp_id),
        "window_scores": str(args.window_scores),
        "truth_gene_list": str(args.truth_gene_list),
        "score_reports": score_reports,
        "best_overlap_record": best,
    }
    (out_dir / "overlap_evaluation_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(str(out_dir / "overlap_evaluation_summary.tsv"))
    print(str(out_dir / "overlap_evaluation_summary.json"))


if __name__ == "__main__":
    main()
