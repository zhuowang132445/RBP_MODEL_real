from __future__ import annotations

import pandas as pd


def build_baseline_score_table(
    gene_scores: pd.DataFrame,
    score_name: str = "inverted_base_motif_structure_score",
    aggregation: str = "top3_mean_score",
) -> pd.DataFrame:
    frame = gene_scores.copy()
    frame["gene_id"] = frame["gene_id"].astype(str)
    # The input gene table is already the aggregation of score_name by aggregation.
    frame["baseline_score"] = pd.to_numeric(frame[aggregation], errors="coerce")
    frame["binding_potential_score"] = frame["baseline_score"]
    frame["baseline_score_name"] = score_name
    frame["baseline_aggregation"] = aggregation
    frame = frame.sort_values(["baseline_score", "gene_id"], ascending=[False, True], na_position="last").reset_index(drop=True)
    frame["baseline_rank"] = range(1, len(frame) + 1)
    return frame
