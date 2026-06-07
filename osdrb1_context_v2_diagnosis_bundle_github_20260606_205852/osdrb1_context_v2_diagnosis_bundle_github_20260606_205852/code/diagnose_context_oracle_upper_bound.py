#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rbp_trace.config import load_config
from rbp_trace.evaluation.overlap_eval import evaluate_topk
from rbp_trace.io_utils import ensure_dir, ensure_path_exists, read_table, write_lines
from rbp_trace.modules.hairpin_stem_loop import compute_paired_architecture_features
from rbp_trace.modules.region_annotation import annotate_region_features, load_region_model, summarize_region_top3
from rbp_trace.modules.repeat_inverted_repeat import compute_repeat_features
from rbp_trace.scoring.baseline_scores import build_baseline_score_table
from rbp_trace.scoring.score_normalization import robust_zscore


TOP_WINDOW_KS = [3, 10, 20]


def _compute_repeat_row(payload: tuple[str, int, str]) -> dict:
    gene_id, window_rank, seq = payload
    return {"gene_id": gene_id, "window_rank": window_rank, **compute_repeat_features(seq)}


def load_truth_gene_ids(path: str | Path) -> set[str]:
    frame = read_table(path)
    if "gene_id" not in frame.columns:
        raise ValueError(f"truth gene list missing gene_id column: {path}")
    return {str(x) for x in frame["gene_id"].dropna().astype(str).tolist()}


def merge_top_windows(existing: list[dict], new_items: list[dict], cap: int) -> list[dict]:
    combined = existing + new_items
    combined.sort(key=lambda item: (-float(item["baseline_window_score"]), str(item["transcript_id"]), int(item["window_start"])))
    return combined[:cap]


def collect_top_windows(window_score_path: Path, top_k: int, chunksize: int = 200000) -> pd.DataFrame:
    usecols = [
        "gene_id",
        "transcript_id",
        "window_start",
        "window_end",
        "rna_seq",
        "inverted_base_motif_structure_score",
        "paired_probability_mean",
        "fraction_high_paired",
        "base_score",
    ]
    top_windows: dict[str, list[dict]] = {}
    for chunk in pd.read_csv(window_score_path, sep="\t", compression="gzip", usecols=usecols, chunksize=chunksize):
        chunk["gene_id"] = chunk["gene_id"].astype(str)
        chunk["transcript_id"] = chunk["transcript_id"].astype(str)
        top_chunk = (
            chunk.sort_values(
                ["gene_id", "inverted_base_motif_structure_score", "transcript_id", "window_start"],
                ascending=[True, False, True, True],
            )
            .groupby("gene_id", sort=False)
            .head(top_k)
        )
        for gene_id, sub in top_chunk.groupby("gene_id", sort=False):
            items = [
                {
                    "gene_id": str(row.gene_id),
                    "transcript_id": str(row.transcript_id),
                    "window_start": int(row.window_start),
                    "window_end": int(row.window_end),
                    "rna_seq": str(row.rna_seq).upper().replace("T", "U"),
                    "baseline_window_score": float(row.inverted_base_motif_structure_score),
                    "paired_probability_mean": float(row.paired_probability_mean),
                    "fraction_high_paired": float(row.fraction_high_paired),
                    "base_score": float(row.base_score),
                }
                for row in sub.itertuples(index=False)
            ]
            top_windows[str(gene_id)] = merge_top_windows(top_windows.get(str(gene_id), []), items, cap=top_k)
    rows = []
    for gene_id, items in top_windows.items():
        for rank, item in enumerate(items, start=1):
            rows.append({"gene_id": gene_id, "window_rank": rank, **item})
    return pd.DataFrame(rows)


def build_window_row_index(window_table: Path, target_keys: set[tuple[str, int, int]]) -> dict[tuple[str, int, int], int]:
    matched: dict[tuple[str, int, int], int] = {}
    row_offset = 0
    for chunk in pd.read_csv(window_table, sep="\t", compression="gzip", usecols=["transcript_id", "window_start", "window_end"], chunksize=200000):
        for row in chunk.itertuples(index=False):
            key = (str(row.transcript_id), int(row.window_start), int(row.window_end))
            if key in target_keys and key not in matched:
                matched[key] = row_offset
            row_offset += 1
    return matched


def compute_window_module_features(
    top_windows: pd.DataFrame,
    config,
    diagnosis_dir: Path,
    workers: int = 1,
) -> pd.DataFrame:
    region_windows = annotate_region_features(
        windows=top_windows[["gene_id", "transcript_id", "window_start", "window_end", "window_rank"]].copy(),
        gtf_path=config.paths.get("annotation_gtf"),
        out_path=diagnosis_dir / "region_annotation_features.tsv",
        missing_report_path=diagnosis_dir / "missing_region_annotation_report.txt",
    )

    repeat_payloads = [(str(row.gene_id), int(row.window_rank), str(row.rna_seq)) for row in top_windows.itertuples(index=False)]
    if workers > 1:
        with get_context("fork").Pool(processes=workers) as pool:
            repeat_rows = list(pool.imap(_compute_repeat_row, repeat_payloads, chunksize=512))
    else:
        repeat_rows = [_compute_repeat_row(item) for item in repeat_payloads]
    repeat_df = pd.DataFrame(repeat_rows)

    target_keys = {(str(r.transcript_id), int(r.window_start), int(r.window_end)) for r in top_windows.itertuples(index=False)}
    row_index = build_window_row_index(ensure_path_exists(config.paths["window_table"], "window table"), target_keys)
    paired_matrix = np.load(ensure_path_exists(config.paths["structure_npy"], "structure npy"), mmap_mode="r")
    paired_rows = []
    for row in top_windows.itertuples(index=False):
        key = (str(row.transcript_id), int(row.window_start), int(row.window_end))
        vec = paired_matrix[row_index[key]][: len(row.rna_seq)] if key in row_index else np.asarray([], dtype=float)
        paired_rows.append({"gene_id": row.gene_id, "window_rank": row.window_rank, **compute_paired_architecture_features(vec)})
    paired_df = pd.DataFrame(paired_rows)

    merged = (
        top_windows.merge(region_windows, on=["gene_id", "transcript_id", "window_start", "window_end", "window_rank"], how="left")
        .merge(repeat_df, on=["gene_id", "window_rank"], how="left")
        .merge(paired_df, on=["gene_id", "window_rank"], how="left")
    )
    merged.to_csv(diagnosis_dir / "top20_window_module_features.tsv", sep="\t", index=False)
    return merged


def aggregate_gene_features_for_k(top_windows: pd.DataFrame, baseline_gene: pd.DataFrame, top_k: int) -> pd.DataFrame:
    rows = []
    for gene_id, sub in top_windows[top_windows["window_rank"] <= top_k].groupby("gene_id", sort=False):
        def num_mean(col: str) -> float:
            return float(pd.to_numeric(sub[col], errors="coerce").mean())

        def num_max(col: str) -> float:
            return float(pd.to_numeric(sub[col], errors="coerce").max())

        rows.append(
            {
                "gene_id": str(gene_id),
                "top_window_k": int(top_k),
                "best_window_region_type": str(sub.iloc[0]["region_type"]),
                "best_window_inferred_region_type": str(sub.iloc[0]["inferred_region_type"]),
                "topk_CDS_fraction": float(pd.to_numeric(sub["is_CDS"], errors="coerce").mean()),
                "topk_UTR_fraction": float((pd.to_numeric(sub["is_5UTR"], errors="coerce") + pd.to_numeric(sub["is_3UTR"], errors="coerce")).mean()),
                "topk_ncRNA_fraction": float(pd.to_numeric(sub["is_ncRNA"], errors="coerce").mean()),
                "topk_unknown_fraction": float(pd.to_numeric(sub["is_unknown"], errors="coerce").mean()),
                "topk_inferred_CDS_fraction": float(pd.to_numeric(sub["inferred_is_CDS"], errors="coerce").mean()),
                "topk_inferred_UTR_fraction": float((pd.to_numeric(sub["inferred_is_5UTR"], errors="coerce") + pd.to_numeric(sub["inferred_is_3UTR"], errors="coerce")).mean()),
                "topk_inferred_ncRNA_fraction": float(pd.to_numeric(sub["inferred_is_ncRNA"], errors="coerce").mean()),
                "topk_inferred_unknown_fraction": float(pd.to_numeric(sub["inferred_is_unknown"], errors="coerce").mean()),
                "topk_self_complementarity_score_mean": num_mean("self_complementarity_score"),
                "topk_inverted_repeat_score_mean": num_mean("inverted_repeat_score"),
                "topk_simple_repeat_score_mean": num_mean("simple_repeat_score"),
                "topk_simple_repeat_penalty_score_mean": num_mean("simple_repeat_penalty_score"),
                "topk_reverse_complement_4mer_density_mean": num_mean("reverse_complement_4mer_density"),
                "topk_reverse_complement_5mer_density_mean": num_mean("reverse_complement_5mer_density"),
                "topk_reverse_complement_6mer_density_mean": num_mean("reverse_complement_6mer_density"),
                "topk_paired_architecture_score_mean": num_mean("paired_architecture_score"),
                "topk_stem_like_score_mean": num_mean("stem_like_score"),
                "topk_paired_run_density_mean": num_mean("paired_run_density"),
                "topk_max_local_paired_mean_31nt_mean": num_mean("max_local_paired_mean_31nt"),
                "topk_max_local_paired_mean_41nt_mean": num_mean("max_local_paired_mean_41nt"),
                "topk_max_local_paired_mean_81nt_mean": num_mean("max_local_paired_mean_81nt"),
                "topk_longest_high_paired_run_mean": num_mean("longest_high_paired_run"),
                "max_paired_architecture_score": num_max("paired_architecture_score"),
                "fraction_high_paired_architecture_windows": float(pd.to_numeric(sub["is_high_paired_architecture_window"], errors="coerce").mean()),
            }
        )
    agg = pd.DataFrame(rows)
    out = baseline_gene.merge(agg, on="gene_id", how="left")
    return out


def build_candidate_table(gene_frame: pd.DataFrame, top_k: int, betas: list[float]) -> tuple[pd.DataFrame, list[dict]]:
    frame = gene_frame.copy()
    baseline_z = robust_zscore(frame["baseline_score"])
    region_explicit = (
        robust_zscore(frame["topk_UTR_fraction"]) + 0.25 * robust_zscore(frame["topk_ncRNA_fraction"]) - 0.5 * robust_zscore(frame["topk_CDS_fraction"])
    ) / 1.75
    region_inferred = (
        robust_zscore(frame["topk_inferred_UTR_fraction"]) + 0.25 * robust_zscore(frame["topk_inferred_ncRNA_fraction"]) - 0.5 * robust_zscore(frame["topk_inferred_CDS_fraction"])
    ) / 1.75
    self_comp = robust_zscore(frame["topk_self_complementarity_score_mean"])
    inv_repeat = robust_zscore(frame["topk_inverted_repeat_score_mean"])
    simple_penalty = robust_zscore(frame["topk_simple_repeat_penalty_score_mean"])
    paired_arch = (
        robust_zscore(frame["topk_paired_architecture_score_mean"])
        + robust_zscore(frame["max_paired_architecture_score"])
        + robust_zscore(frame["fraction_high_paired_architecture_windows"])
        + robust_zscore(frame["topk_max_local_paired_mean_41nt_mean"])
        + robust_zscore(frame["topk_max_local_paired_mean_81nt_mean"])
    ) / 5.0

    base_cols = {
        "baseline_score_z": baseline_z,
        "region_context_score_explicit": region_explicit,
        "region_context_score_inferred": region_inferred,
        "self_complementarity_context_score": self_comp,
        "inverted_repeat_context_score": inv_repeat,
        "simple_repeat_penalty_score": simple_penalty,
        "paired_architecture_context_score": paired_arch,
    }
    candidate_cols: dict[str, pd.Series] = {}

    candidate_meta: list[dict] = []

    def add_candidate(name: str, series: pd.Series, family: str, module_variant: str = "", region_mode: str = "", beta_region: float | None = None, beta_repeat: float | None = None, beta_hairpin: float | None = None) -> None:
        candidate_cols[name] = series
        candidate_meta.append(
            {
                "candidate_name": name,
                "family": family,
                "module_variant": module_variant,
                "region_mode": region_mode,
                "top_window_k": int(top_k),
                "beta_region": beta_region,
                "beta_repeat": beta_repeat,
                "beta_hairpin": beta_hairpin,
            }
        )

    add_candidate("baseline_score", score_table["baseline_score"], family="baseline")
    add_candidate(f"region_context_score_explicit__k_{top_k}", score_table["region_context_score_explicit"], family="region_raw", region_mode="explicit")
    add_candidate(f"region_context_score_inferred__k_{top_k}", score_table["region_context_score_inferred"], family="region_raw", region_mode="inferred")
    add_candidate(f"self_complementarity_context_score__k_{top_k}", score_table["self_complementarity_context_score"], family="self_complementarity_raw", module_variant="self_complementarity")
    add_candidate(f"inverted_repeat_context_score__k_{top_k}", score_table["inverted_repeat_context_score"], family="self_complementarity_raw", module_variant="inverted_repeat")
    add_candidate(f"paired_architecture_context_score__k_{top_k}", score_table["paired_architecture_context_score"], family="paired_architecture_raw", module_variant="paired_architecture")

    repeat_variants = {
        "self_complementarity": self_comp,
        "inverted_repeat": inv_repeat,
        "minus_simple_repeat_penalty": -simple_penalty,
        "self_complementarity_minus_simple_repeat_penalty": self_comp - simple_penalty,
    }
    region_variants = {"explicit": region_explicit, "inferred": region_inferred}

    for beta in betas:
        for region_mode, region_score in region_variants.items():
            add_candidate(
                f"rbp_trace_v2_score_region__{region_mode}__k_{top_k}__beta_{beta}",
                baseline_z + beta * region_score,
                family="region",
                region_mode=region_mode,
                beta_region=float(beta),
            )
        for variant_name, repeat_score in repeat_variants.items():
            add_candidate(
                f"rbp_trace_v2_score_repeat__{variant_name}__k_{top_k}__beta_{beta}",
                baseline_z + beta * repeat_score,
                family="self_complementarity",
                module_variant=variant_name,
                beta_repeat=float(beta),
            )
        add_candidate(
            f"rbp_trace_v2_score_paired_architecture__k_{top_k}__beta_{beta}",
            baseline_z + beta * paired_arch,
            family="paired_architecture",
            module_variant="paired_architecture",
            beta_hairpin=float(beta),
        )

    for beta_region in betas:
        for beta_repeat in betas:
            for beta_hairpin in betas:
                for repeat_name, repeat_score in repeat_variants.items():
                    add_candidate(
                        f"rbp_trace_v2_score_all_context__k_{top_k}__beta_region_{beta_region}__beta_repeat_{beta_repeat}__beta_hairpin_{beta_hairpin}__repeat_{repeat_name}",
                        baseline_z + beta_region * region_inferred + beta_repeat * repeat_score + beta_hairpin * paired_arch,
                        family="all_context",
                        module_variant=repeat_name,
                        region_mode="inferred",
                        beta_region=float(beta_region),
                        beta_repeat=float(beta_repeat),
                        beta_hairpin=float(beta_hairpin),
                    )

    score_table = pd.concat(
        [
            frame[["gene_id", "baseline_score"]].copy(),
            pd.DataFrame(base_cols),
            pd.DataFrame(candidate_cols),
        ],
        axis=1,
    )
    return score_table, candidate_meta


def evaluate_candidate_grid(score_table: pd.DataFrame, candidate_meta: list[dict], truth_gene_ids: set[str], top_ks: list[int]) -> pd.DataFrame:
    rows = []
    for meta in candidate_meta:
        eval_df = evaluate_topk(score_table[["gene_id", meta["candidate_name"]]].rename(columns={meta["candidate_name"]: "score"}), "score", truth_gene_ids, top_ks, label=meta["candidate_name"])
        for row in eval_df.itertuples(index=False):
            rows.append(
                {
                    **meta,
                    "top_k": int(row.top_k),
                    "universe_genes": int(row.universe_genes),
                    "truth_genes": int(row.truth_genes),
                    "overlap": int(row.overlap),
                    "precision": float(row.precision),
                    "recall": float(row.recall),
                    "expected_random": float(row.expected_random),
                    "fold_enrichment": float(row.fold_enrichment),
                    "hypergeom_pvalue": float(row.hypergeom_pvalue),
                }
            )
    return pd.DataFrame(rows)


def build_region_diagnostic(
    region_model: dict[str, dict],
    top_windows: pd.DataFrame,
    agg_top3: pd.DataFrame,
    baseline_gene: pd.DataFrame,
    truth_gene_ids: set[str],
) -> pd.DataFrame:
    rows = []
    n_transcripts_in_gtf = len(region_model)
    window_transcripts = set(top_windows["transcript_id"].astype(str))
    matched = sum(1 for tx in window_transcripts if tx in region_model)
    rows.extend(
        [
            {"section": "summary", "metric": "n_transcripts_in_gtf", "group": "all", "value": n_transcripts_in_gtf},
            {"section": "summary", "metric": "n_window_transcripts", "group": "all", "value": len(window_transcripts)},
            {"section": "summary", "metric": "n_matched_transcripts", "group": "all", "value": matched},
            {"section": "summary", "metric": "match_rate", "group": "all", "value": matched / len(window_transcripts) if window_transcripts else np.nan},
        ]
    )

    for region_col, label in [("region_type", "original"), ("inferred_region_type", "inferred")]:
        counts = top_windows[region_col].astype(str).value_counts()
        for key, value in counts.items():
            rows.append({"section": "region_counts", "metric": label, "group": key, "value": int(value)})

    for col in ["top3_CDS_fraction", "top3_UTR_fraction", "top3_ncRNA_fraction", "top3_inferred_CDS_fraction", "top3_inferred_UTR_fraction", "top3_inferred_ncRNA_fraction"]:
        vals = pd.to_numeric(agg_top3[col], errors="coerce")
        rows.extend(
            [
                {"section": "distribution", "metric": col, "group": "mean", "value": float(vals.mean())},
                {"section": "distribution", "metric": col, "group": "median", "value": float(vals.median())},
                {"section": "distribution", "metric": col, "group": "nonzero_count", "value": int((vals.fillna(0) > 0).sum())},
            ]
        )

    baseline_ranked = baseline_gene.sort_values(["baseline_score", "gene_id"], ascending=[False, True]).reset_index(drop=True)
    comparable_truth = set(baseline_ranked["gene_id"]) & truth_gene_ids
    top200 = set(baseline_ranked.head(200)["gene_id"])
    groups = {
        "Top200_TP": top200 & comparable_truth,
        "Top200_FP": top200 - comparable_truth,
        "truth_outside_Top200": comparable_truth - top200,
    }
    region_map = agg_top3.set_index("gene_id")
    for group_name, gene_ids in groups.items():
        for col, label in [("best_window_region_type", "original"), ("best_window_inferred_region_type", "inferred")]:
            counts = region_map.loc[list(gene_ids), col].astype(str).value_counts() if gene_ids else pd.Series(dtype=int)
            total = counts.sum()
            for key, value in counts.items():
                rows.append({"section": "top200_region_distribution", "metric": f"{group_name}__{label}", "group": key, "value": float(value / total) if total else np.nan})
    return pd.DataFrame(rows)


def summarize_oracle(grid: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    top200 = grid[grid["top_k"] == 200].copy()

    def best_row(sub: pd.DataFrame) -> pd.Series:
        sub = sub.sort_values(["overlap", "fold_enrichment", "hypergeom_pvalue"], ascending=[False, False, True]).reset_index(drop=True)
        return sub.iloc[0]

    families = {
        "baseline": top200[top200["family"] == "baseline"],
        "best_region": top200[top200["family"] == "region"],
        "best_self_complementarity": top200[top200["family"] == "self_complementarity"],
        "best_paired_architecture": top200[top200["family"] == "paired_architecture"],
        "best_all_context": top200[top200["family"] == "all_context"],
        "best_overall": top200,
    }
    rows = []
    for label, sub in families.items():
        if sub.empty:
            continue
        row = best_row(sub)
        rows.append(
            {
                "summary_label": label,
                "candidate_name": row["candidate_name"],
                "family": row["family"],
                "module_variant": row["module_variant"],
                "region_mode": row["region_mode"],
                "top_window_k": int(row["top_window_k"]),
                "beta_region": row["beta_region"],
                "beta_repeat": row["beta_repeat"],
                "beta_hairpin": row["beta_hairpin"],
                "top200_overlap": int(row["overlap"]),
                "top200_precision": float(row["precision"]),
                "top200_recall": float(row["recall"]),
                "top200_fold_enrichment": float(row["fold_enrichment"]),
                "top200_hypergeom_pvalue": float(row["hypergeom_pvalue"]),
            }
        )
    summary = pd.DataFrame(rows)

    comparison_rows = []
    for k in TOP_WINDOW_KS:
        sub = top200[top200["top_window_k"] == k]
        for family in ["region", "self_complementarity", "paired_architecture", "all_context"]:
            fam = sub[sub["family"] == family]
            if fam.empty:
                continue
            row = best_row(fam)
            comparison_rows.append(
                {
                    "top_window_k": int(k),
                    "family": family,
                    "best_candidate_name": row["candidate_name"],
                    "module_variant": row["module_variant"],
                    "region_mode": row["region_mode"],
                    "beta_region": row["beta_region"],
                    "beta_repeat": row["beta_repeat"],
                    "beta_hairpin": row["beta_hairpin"],
                    "top200_overlap": int(row["overlap"]),
                    "top200_precision": float(row["precision"]),
                    "top200_fold_enrichment": float(row["fold_enrichment"]),
                    "top200_hypergeom_pvalue": float(row["hypergeom_pvalue"]),
                }
            )
    return summary, pd.DataFrame(comparison_rows)


def build_report(
    diagnosis_dir: Path,
    baseline_full: pd.DataFrame,
    current_heldout: pd.DataFrame,
    oracle_summary: pd.DataFrame,
    region_diag: pd.DataFrame,
    top_window_comparison: pd.DataFrame,
) -> None:
    baseline_top200 = baseline_full[baseline_full["top_k"] == 200].iloc[0]
    baseline_top500 = baseline_full[baseline_full["top_k"] == 500].iloc[0]
    baseline_top1000 = baseline_full[baseline_full["top_k"] == 1000].iloc[0]
    current_all_context_top200 = current_heldout[(current_heldout["score_family"] == "rbp_trace_v2_score_all_context") & (current_heldout["top_k"] == 200)]["overlap"].mean()

    oracle_best = oracle_summary.set_index("summary_label")
    best_region = oracle_best.loc["best_region"] if "best_region" in oracle_best.index else None
    best_self = oracle_best.loc["best_self_complementarity"] if "best_self_complementarity" in oracle_best.index else None
    best_paired = oracle_best.loc["best_paired_architecture"] if "best_paired_architecture" in oracle_best.index else None
    best_all = oracle_best.loc["best_all_context"] if "best_all_context" in oracle_best.index else None
    best_overall = oracle_best.loc["best_overall"] if "best_overall" in oracle_best.index else None

    utr_nonzero = region_diag[(region_diag["section"] == "distribution") & (region_diag["metric"] == "top3_UTR_fraction") & (region_diag["group"] == "nonzero_count")]["value"].iloc[0]
    inferred_utr_nonzero = region_diag[(region_diag["section"] == "distribution") & (region_diag["metric"] == "top3_inferred_UTR_fraction") & (region_diag["group"] == "nonzero_count")]["value"].iloc[0]
    best_topk_rows = top_window_comparison.sort_values(["family", "top200_overlap", "top200_fold_enrichment"], ascending=[True, False, False]).drop_duplicates(["family"])

    lines = [
        "# RBP_TRACE_V2_1 CONTEXT DIAGNOSIS REPORT",
        "",
        "## Baseline",
        "",
        f"- baseline Top200: {int(baseline_top200['overlap'])}",
        f"- baseline Top500: {int(baseline_top500['overlap'])}",
        f"- baseline Top1000: {int(baseline_top1000['overlap'])}",
        "",
        "## Current V2 Held-out",
        "",
        f"- current V2 selected all_context held-out Top200 mean overlap: {current_all_context_top200:.2f}",
        "",
        "## Oracle Upper Bound",
        "",
    ]
    if best_region is not None:
        lines.append(f"- best region Top200: {int(best_region['top200_overlap'])} via `{best_region['candidate_name']}`")
    if best_self is not None:
        lines.append(f"- best self-complementarity Top200: {int(best_self['top200_overlap'])} via `{best_self['candidate_name']}`")
    if best_paired is not None:
        lines.append(f"- best paired-architecture Top200: {int(best_paired['top200_overlap'])} via `{best_paired['candidate_name']}`")
    if best_all is not None:
        lines.append(f"- best all_context Top200: {int(best_all['top200_overlap'])} via `{best_all['candidate_name']}`")
    if best_overall is not None:
        lines.append(f"- best overall Top200: {int(best_overall['top200_overlap'])} via `{best_overall['candidate_name']}`")

    lines.extend(["", "## Region Diagnostic", ""])
    if utr_nonzero == 0 and inferred_utr_nonzero > 0:
        lines.append("- original region annotation is effectively missing UTR signal; inferred UTR from CDS boundaries partially restores it.")
    elif utr_nonzero == 0:
        lines.append("- region annotation still has no usable UTR signal after inference.")
    else:
        lines.append("- original region annotation already contains usable UTR signal.")

    lines.extend(["", "## Module Contribution", ""])
    if best_self is not None:
        lines.append(f"- self-complementarity module oracle Top200 upper bound: {int(best_self['top200_overlap'])}")
    if best_paired is not None:
        lines.append(f"- paired-architecture module oracle Top200 upper bound: {int(best_paired['top200_overlap'])}")
    if best_paired is not None and best_self is not None:
        if best_paired["top200_overlap"] >= best_self["top200_overlap"]:
            lines.append("- paired-architecture remains the dominant single context module.")
        else:
            lines.append("- self-complementarity has a stronger single-module oracle contribution than paired-architecture.")

    lines.extend(["", "## Top-window-k Diagnostic", ""])
    for row in best_topk_rows.itertuples(index=False):
        lines.append(f"- {row.family}: best top-window-k = {int(row.top_window_k)}, Top200 = {int(row.top200_overlap)}")

    lines.extend(
        [
            "",
            "## Reality Check",
            "",
            "- Full-truth oracle results are upper-bound diagnostics, not generalization estimates.",
            "- Realistic current ceiling should be interpreted between the current held-out all_context result and the oracle best candidate.",
        ]
    )
    if best_all is not None:
        lines.append(f"- Current three-module ceiling estimate for OsDRB1 is roughly held-out Top200 {current_all_context_top200:.1f} versus oracle Top200 {int(best_all['top200_overlap'])}.")
    write_lines(diagnosis_dir / "RBP_TRACE_V2_1_CONTEXT_DIAGNOSIS_REPORT.md", lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--top-window-ks", default="3,10,20")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    config = load_config(args.config)
    diagnosis_dir = ensure_dir(config.results_root / "osdrb1_context_v2_diagnosis")
    top_window_ks = [int(x) for x in str(args.top_window_ks).split(",") if x.strip()]
    max_top_k = max(top_window_ks)
    betas = [float(x) for x in config.context_betas]

    baseline_gene = build_baseline_score_table(read_table(ensure_path_exists(config.paths["gene_score_table"], "gene score table")))
    truth_gene_ids = load_truth_gene_ids(ensure_path_exists(config.paths["truth_gene_list"], "truth gene list"))
    baseline_full = evaluate_topk(baseline_gene, "baseline_score", truth_gene_ids, config.top_ks, label="baseline_score")

    top_windows = collect_top_windows(ensure_path_exists(config.paths["window_score_table"], "window score table"), max_top_k)
    top_windows = compute_window_module_features(top_windows, config, diagnosis_dir, workers=max(int(args.workers), 1))

    region_model = load_region_model(ensure_path_exists(config.paths["annotation_gtf"], "annotation gtf"))
    agg_top3 = summarize_region_top3(top_windows[top_windows["window_rank"] <= 3].copy())
    region_diag = build_region_diagnostic(region_model, top_windows, agg_top3, baseline_gene, truth_gene_ids)
    region_diag.to_csv(diagnosis_dir / "region_annotation_diagnostic.tsv", sep="\t", index=False)

    all_grid_rows = []
    top_window_summary_tables = []
    for top_k in top_window_ks:
        gene_frame = aggregate_gene_features_for_k(top_windows, baseline_gene[["gene_id", "transcript_id", "baseline_score", "baseline_rank", "n_windows"]].copy(), top_k)
        gene_frame.to_csv(diagnosis_dir / f"gene_features_top{top_k}.tsv", sep="\t", index=False)
        score_table, meta = build_candidate_table(gene_frame, top_k, betas)
        score_table.to_csv(diagnosis_dir / f"context_score_table_top{top_k}.tsv", sep="\t", index=False)
        grid = evaluate_candidate_grid(score_table, meta, truth_gene_ids, config.top_ks)
        all_grid_rows.append(grid)

    grid_all = pd.concat(all_grid_rows, ignore_index=True)
    grid_all.to_csv(diagnosis_dir / "context_candidate_grid_full_truth_oracle.tsv", sep="\t", index=False)
    oracle_summary, top_window_comparison = summarize_oracle(grid_all)
    oracle_summary.to_csv(diagnosis_dir / "context_oracle_upper_bound_summary.tsv", sep="\t", index=False)
    top_window_comparison.to_csv(diagnosis_dir / "top_window_k_comparison.tsv", sep="\t", index=False)

    current_heldout = read_table(config.results_root / "osdrb1_context_v2" / "context_v2_heldout_summary.tsv")
    build_report(diagnosis_dir, baseline_full, current_heldout, oracle_summary, region_diag, top_window_comparison)


if __name__ == "__main__":
    main()
