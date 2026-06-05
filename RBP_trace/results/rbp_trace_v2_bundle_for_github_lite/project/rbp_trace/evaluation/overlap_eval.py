from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from scipy.stats import hypergeom
except Exception:  # pragma: no cover
    hypergeom = None


def safe_hypergeom_sf(k_minus_one: int, population: int, truth_count: int, draws: int) -> float:
    if hypergeom is not None:
        return float(hypergeom.sf(k_minus_one, population, truth_count, draws))
    denom = math.comb(population, draws)
    total = 0.0
    for k in range(max(0, k_minus_one + 1), min(truth_count, draws) + 1):
        total += (math.comb(truth_count, k) * math.comb(population - truth_count, draws - k)) / denom
    return float(min(max(total, 0.0), 1.0))


def evaluate_topk(
    ranked: pd.DataFrame,
    score_col: str,
    truth_gene_ids: set[str],
    top_ks: Iterable[int],
    label: str,
    seed: int | None = None,
) -> pd.DataFrame:
    ranked = ranked.copy()
    ranked["gene_id"] = ranked["gene_id"].astype(str)
    ranked = ranked.sort_values([score_col, "gene_id"], ascending=[False, True], na_position="last").reset_index(drop=True)
    universe = set(ranked["gene_id"])
    comparable_truth = universe & set(truth_gene_ids)
    population = len(universe)
    truth_count = len(comparable_truth)
    rows = []
    for top_k in top_ks:
        subset = ranked.head(min(int(top_k), len(ranked)))
        overlap = int(subset["gene_id"].isin(comparable_truth).sum())
        expected = (len(subset) * truth_count / population) if population else float("nan")
        precision = (overlap / len(subset)) if len(subset) else float("nan")
        recall = (overlap / truth_count) if truth_count else float("nan")
        fold = (overlap / expected) if expected and np.isfinite(expected) and expected > 0 else float("nan")
        pvalue = safe_hypergeom_sf(overlap - 1, population, truth_count, len(subset)) if population and truth_count and len(subset) else float("nan")
        rows.append(
            {
                "label": label,
                "seed": seed,
                "top_k": int(top_k),
                "universe_genes": int(population),
                "truth_genes": int(truth_count),
                "overlap": int(overlap),
                "precision": float(precision),
                "recall": float(recall),
                "expected_random": float(expected),
                "fold_enrichment": float(fold),
                "hypergeom_pvalue": float(pvalue),
            }
        )
    return pd.DataFrame(rows)
