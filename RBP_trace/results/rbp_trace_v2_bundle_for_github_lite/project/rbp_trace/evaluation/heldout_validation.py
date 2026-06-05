from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .overlap_eval import evaluate_topk


@dataclass
class SplitSpec:
    seed: int
    calibration_truth: set[str]
    heldout_truth: set[str]


def make_splits(truth_gene_ids: set[str], n_splits: int, calibration_fraction: float, seed0: int) -> list[SplitSpec]:
    comparable_truth = sorted(set(truth_gene_ids))
    rng_master = np.random.default_rng(int(seed0))
    splits: list[SplitSpec] = []
    for _ in range(int(n_splits)):
        seed = int(rng_master.integers(0, 2**31 - 1))
        rng = np.random.default_rng(seed)
        shuffled = np.asarray(comparable_truth, dtype=object).copy()
        rng.shuffle(shuffled)
        n_cal = max(1, int(round(len(shuffled) * float(calibration_fraction))))
        n_cal = min(n_cal, len(shuffled) - 1)
        calibration_truth = set(shuffled[:n_cal].tolist())
        heldout_truth = set(shuffled[n_cal:].tolist())
        splits.append(SplitSpec(seed=seed, calibration_truth=calibration_truth, heldout_truth=heldout_truth))
    return splits


def evaluate_fixed_score(
    frame: pd.DataFrame,
    score_col: str,
    splits: Iterable[SplitSpec],
    top_ks: list[int],
    label: str,
) -> pd.DataFrame:
    rows = []
    for split in splits:
        rows.append(evaluate_topk(frame, score_col, split.heldout_truth, top_ks, label=label, seed=split.seed))
    return pd.concat(rows, ignore_index=True)


def choose_best_beta(
    frame: pd.DataFrame,
    candidate_scores: dict[str, pd.Series],
    split: SplitSpec,
    top_ks: list[int],
) -> tuple[str, pd.DataFrame]:
    best_name = ""
    best_tuple = None
    summary_rows = []
    for name, series in candidate_scores.items():
        tmp = frame[["gene_id"]].copy()
        tmp["candidate_score"] = series
        eval_df = evaluate_topk(tmp, "candidate_score", split.calibration_truth, top_ks, label=name, seed=split.seed)
        top200 = eval_df[eval_df["top_k"] == 200].iloc[0]
        top500 = eval_df[eval_df["top_k"] == 500].iloc[0]
        metric = (
            -np.log10(max(float(top200["hypergeom_pvalue"]), 1e-300)),
            float(top200["overlap"]),
            float(top200["fold_enrichment"]),
            float(top500["overlap"]),
        )
        summary_rows.append(
            {
                "seed": split.seed,
                "candidate_name": name,
                "top200_overlap": int(top200["overlap"]),
                "top200_fold_enrichment": float(top200["fold_enrichment"]),
                "top200_hypergeom_pvalue": float(top200["hypergeom_pvalue"]),
                "top500_overlap": int(top500["overlap"]),
                "top500_hypergeom_pvalue": float(top500["hypergeom_pvalue"]),
            }
        )
        if best_tuple is None or metric > best_tuple:
            best_tuple = metric
            best_name = name
    return best_name, pd.DataFrame(summary_rows)
