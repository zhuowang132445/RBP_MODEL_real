from __future__ import annotations

from typing import Iterable

import pandas as pd


def assign_tp_fp_groups(frame: pd.DataFrame, score_col: str, truth_gene_ids: set[str], top_k: int) -> pd.DataFrame:
    ranked = frame.sort_values([score_col, "gene_id"], ascending=[False, True], na_position="last").reset_index(drop=True).copy()
    ranked["gene_id"] = ranked["gene_id"].astype(str)
    top = ranked.head(int(top_k))
    comparable_truth = set(ranked["gene_id"]) & set(truth_gene_ids)
    ranked["audit_group"] = "background"
    ranked.loc[ranked["gene_id"].isin(set(top["gene_id"]) & comparable_truth), "audit_group"] = f"Top{top_k} true positives"
    ranked.loc[ranked["gene_id"].isin(set(top["gene_id"]) - comparable_truth), "audit_group"] = f"Top{top_k} false positives"
    ranked.loc[ranked["gene_id"].isin(comparable_truth - set(top["gene_id"])), "audit_group"] = f"truth outside Top{top_k}"
    return ranked


def summarize_groups(frame: pd.DataFrame, numeric_cols: Iterable[str]) -> pd.DataFrame:
    rows = []
    for group_name, sub in frame.groupby("audit_group", sort=False):
        if group_name == "background":
            continue
        row = {"audit_group": group_name, "n_genes": int(len(sub))}
        for col in numeric_cols:
            values = pd.to_numeric(sub[col], errors="coerce")
            row[f"{col}_mean"] = float(values.mean()) if len(values.dropna()) else float("nan")
            row[f"{col}_median"] = float(values.median()) if len(values.dropna()) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)
