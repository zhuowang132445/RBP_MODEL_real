from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from rbp_trace.config import TraceConfig
from rbp_trace.evaluation.heldout_validation import choose_best_beta, evaluate_fixed_score, make_splits
from rbp_trace.evaluation.overlap_eval import evaluate_topk
from rbp_trace.evaluation.tp_fp_audit import assign_tp_fp_groups, summarize_groups
from rbp_trace.io_utils import ensure_dir, ensure_path_exists, read_table, write_lines
from rbp_trace.modules.hairpin_stem_loop import compute_hairpin_features
from rbp_trace.modules.region_annotation import annotate_region_features, summarize_region_top3
from rbp_trace.modules.repeat_inverted_repeat import compute_repeat_features
from rbp_trace.scoring.baseline_scores import build_baseline_score_table
from rbp_trace.scoring.score_normalization import robust_zscore


def _load_truth_gene_ids(path: str | Path) -> set[str]:
    frame = read_table(path)
    if "gene_id" not in frame.columns:
        raise ValueError(f"truth gene list missing gene_id column: {path}")
    return {str(x) for x in frame["gene_id"].dropna().astype(str).tolist()}


def _merge_top_windows(existing: list[dict], new_items: list[dict], cap: int = 3) -> list[dict]:
    combined = existing + new_items
    combined.sort(key=lambda item: (-float(item["baseline_score"]), str(item["transcript_id"]), int(item["window_start"])))
    return combined[:cap]


def _collect_top3_windows(window_score_path: Path, chunksize: int = 200000) -> pd.DataFrame:
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
            .head(3)
        )
        for gene_id, sub in top_chunk.groupby("gene_id", sort=False):
            new_items = [
                {
                    "gene_id": str(row.gene_id),
                    "transcript_id": str(row.transcript_id),
                    "window_start": int(row.window_start),
                    "window_end": int(row.window_end),
                    "rna_seq": str(row.rna_seq).upper().replace("T", "U"),
                    "baseline_score": float(row.inverted_base_motif_structure_score),
                    "paired_probability_mean": float(row.paired_probability_mean),
                    "fraction_high_paired": float(row.fraction_high_paired),
                    "base_score": float(row.base_score),
                }
                for row in sub.itertuples(index=False)
            ]
            top_windows[str(gene_id)] = _merge_top_windows(top_windows.get(str(gene_id), []), new_items)
    rows = []
    for gene_id, items in top_windows.items():
        for rank, item in enumerate(items, start=1):
            rows.append({"gene_id": gene_id, "window_rank": rank, **item})
    return pd.DataFrame(rows)


def _build_window_row_index(window_table: Path, target_keys: set[tuple[str, int, int]]) -> dict[tuple[str, int, int], int]:
    matched: dict[tuple[str, int, int], int] = {}
    row_offset = 0
    for chunk in pd.read_csv(window_table, sep="\t", compression="gzip", usecols=["transcript_id", "window_start", "window_end"], chunksize=200000):
        for row in chunk.itertuples(index=False):
            key = (str(row.transcript_id), int(row.window_start), int(row.window_end))
            if key in target_keys and key not in matched:
                matched[key] = row_offset
            row_offset += 1
    return matched


def build_context_feature_table(
    config: TraceConfig,
    baseline: pd.DataFrame,
    out_dir: Path,
    enable_region: bool,
    enable_repeat: bool,
    enable_hairpin: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    window_score_path = ensure_path_exists(config.paths["window_score_table"], "window score table")
    top_windows = _collect_top3_windows(window_score_path)
    top_windows.to_csv(out_dir / "top3_windows.tsv", sep="\t", index=False)

    windows = top_windows[["gene_id", "transcript_id", "window_start", "window_end", "window_rank"]].copy()
    if enable_region:
        region_windows = annotate_region_features(
            windows=windows,
            gtf_path=config.paths.get("annotation_gtf"),
            out_path=out_dir / "region_annotation_features.tsv",
            missing_report_path=out_dir / "missing_region_annotation_report.txt",
        )
    else:
        region_windows = windows.copy()
        region_windows["region_type"] = "NA"
        for col in ["is_CDS", "is_5UTR", "is_3UTR", "is_ncRNA", "is_unknown"]:
            region_windows[col] = np.nan
        region_windows.to_csv(out_dir / "region_annotation_features.tsv", sep="\t", index=False)
    region_gene = summarize_region_top3(region_windows)

    repeat_rows = []
    for row in top_windows.itertuples(index=False):
        feat = compute_repeat_features(row.rna_seq) if enable_repeat else {k: np.nan for k in [
            "reverse_complement_4mer_density", "reverse_complement_5mer_density", "reverse_complement_6mer_density",
            "longest_internal_rc_match", "internal_rc_pair_count", "palindrome_score", "inverted_repeat_score",
            "local_self_complementarity_score", "simple_repeat_score", "overlap_TE", "TE_class", "TE_family",
            "repeat_overlap_fraction", "distance_to_nearest_TE",
        ]}
        repeat_rows.append({"gene_id": row.gene_id, "window_rank": row.window_rank, **feat})
    repeat_windows = pd.DataFrame(repeat_rows)
    repeat_windows.to_csv(out_dir / "repeat_inverted_repeat_features.tsv", sep="\t", index=False)

    hairpin_rows = []
    structure_path = config.paths.get("structure_npy", "")
    if enable_hairpin and structure_path and Path(structure_path).exists():
        target_keys = {(str(r.transcript_id), int(r.window_start), int(r.window_end)) for r in top_windows.itertuples(index=False)}
        row_index = _build_window_row_index(ensure_path_exists(config.paths["window_table"], "window table"), target_keys)
        paired_matrix = np.load(ensure_path_exists(structure_path, "structure npy"), mmap_mode="r")
        for row in top_windows.itertuples(index=False):
            key = (str(row.transcript_id), int(row.window_start), int(row.window_end))
            vec = paired_matrix[row_index[key]][: len(row.rna_seq)] if key in row_index else np.asarray([], dtype=float)
            feat = compute_hairpin_features(vec)
            hairpin_rows.append({"gene_id": row.gene_id, "window_rank": row.window_rank, **feat})
    else:
        Path(out_dir / "missing_rnafold_report.txt").write_text("RNAfold not required; using paired probability only. Structure input missing or module disabled.\n", encoding="utf-8")
        na_cols = list(compute_hairpin_features(np.asarray([], dtype=float)).keys())
        for row in top_windows.itertuples(index=False):
            hairpin_rows.append({"gene_id": row.gene_id, "window_rank": row.window_rank, **{col: np.nan for col in na_cols}})
    hairpin_windows = pd.DataFrame(hairpin_rows)
    hairpin_windows.to_csv(out_dir / "hairpin_stem_loop_features.tsv", sep="\t", index=False)

    gene_table = baseline.copy()
    region_gene["gene_id"] = region_gene["gene_id"].astype(str)
    gene_table = gene_table.rename(columns={"baseline_score": "baseline_top3_mean_score"})[
        ["gene_id", "transcript_id", "baseline_top3_mean_score", "baseline_rank", "n_windows"]
    ]
    gene_table["gene_id"] = gene_table["gene_id"].astype(str)
    gene_table = gene_table.merge(region_gene, on="gene_id", how="left")

    top_windows["gene_id"] = top_windows["gene_id"].astype(str)
    repeat_windows["gene_id"] = repeat_windows["gene_id"].astype(str)
    hairpin_windows["gene_id"] = hairpin_windows["gene_id"].astype(str)
    top3 = top_windows.merge(repeat_windows, on=["gene_id", "window_rank"], how="left").merge(hairpin_windows, on=["gene_id", "window_rank"], how="left")

    agg_rows = []
    for gene_id, sub in top3.groupby("gene_id", sort=False):
        row = {
            "gene_id": str(gene_id),
            "top3_inverted_repeat_score_mean": float(pd.to_numeric(sub["inverted_repeat_score"], errors="coerce").mean()),
            "top3_palindrome_score_mean": float(pd.to_numeric(sub["palindrome_score"], errors="coerce").mean()),
            "top3_reverse_complement_4mer_density_mean": float(pd.to_numeric(sub["reverse_complement_4mer_density"], errors="coerce").mean()),
            "top3_reverse_complement_5mer_density_mean": float(pd.to_numeric(sub["reverse_complement_5mer_density"], errors="coerce").mean()),
            "top3_reverse_complement_6mer_density_mean": float(pd.to_numeric(sub["reverse_complement_6mer_density"], errors="coerce").mean()),
            "top3_local_self_complementarity_score_mean": float(pd.to_numeric(sub["local_self_complementarity_score"], errors="coerce").mean()),
            "top3_simple_repeat_score_mean": float(pd.to_numeric(sub["simple_repeat_score"], errors="coerce").mean()),
            "top3_stem_like_score_mean": float(pd.to_numeric(sub["stem_like_score"], errors="coerce").mean()),
            "top3_structure_peakiness_score_mean": float(pd.to_numeric(sub["structure_peakiness_score"], errors="coerce").mean()),
            "top3_longest_high_paired_run_mean": float(pd.to_numeric(sub["longest_high_paired_run"], errors="coerce").mean()),
            "top3_paired_run_density_mean": float(pd.to_numeric(sub["paired_run_density"], errors="coerce").mean()),
            "top3_max_local_paired_mean_41nt_mean": float(pd.to_numeric(sub["max_local_paired_mean_41nt"], errors="coerce").mean()),
            "top3_max_local_paired_mean_81nt_mean": float(pd.to_numeric(sub["max_local_paired_mean_81nt"], errors="coerce").mean()),
            "max_stem_like_score": float(pd.to_numeric(sub["stem_like_score"], errors="coerce").max()),
            "max_longest_high_paired_run": float(pd.to_numeric(sub["longest_high_paired_run"], errors="coerce").max()),
        }
        agg_rows.append(row)
    gene_table = gene_table.merge(pd.DataFrame(agg_rows), on="gene_id", how="left")
    gene_table.to_csv(out_dir / "rbp_trace_v2_gene_feature_table.tsv", sep="\t", index=False)
    return gene_table, top3


def _score_family_candidates(feature_table: pd.DataFrame, betas: list[float], enable_region: bool, enable_repeat: bool, enable_hairpin: bool) -> tuple[dict[str, pd.Series], pd.DataFrame]:
    frame = feature_table.copy()
    frame["baseline_score"] = pd.to_numeric(frame["baseline_top3_mean_score"], errors="coerce")
    baseline_score_z = robust_zscore(frame["baseline_score"])
    region_components = [
        robust_zscore(frame["top3_UTR_fraction"]) if "top3_UTR_fraction" in frame else 0.0,
        robust_zscore(frame["top3_ncRNA_fraction"]) if "top3_ncRNA_fraction" in frame else 0.0,
        -robust_zscore(frame["top3_CDS_fraction"]) if "top3_CDS_fraction" in frame else 0.0,
    ]
    region_bonus = sum(region_components) / 3.0 if enable_region else pd.Series(np.zeros(len(frame)), index=frame.index)
    repeat_bonus = (
        robust_zscore(frame["top3_inverted_repeat_score_mean"])
        + robust_zscore(frame["top3_local_self_complementarity_score_mean"])
        + robust_zscore(frame["top3_reverse_complement_5mer_density_mean"])
        + robust_zscore(frame["top3_palindrome_score_mean"])
    ) / 4.0 if enable_repeat else pd.Series(np.zeros(len(frame)), index=frame.index)
    hairpin_bonus = (
        robust_zscore(frame["top3_stem_like_score_mean"])
        + robust_zscore(frame["top3_structure_peakiness_score_mean"])
        + robust_zscore(frame["top3_max_local_paired_mean_41nt_mean"])
        + robust_zscore(frame["top3_paired_run_density_mean"])
    ) / 4.0 if enable_hairpin else pd.Series(np.zeros(len(frame)), index=frame.index)

    score_table = frame[["gene_id", "baseline_score"]].copy()
    score_table["baseline_score_z"] = baseline_score_z
    score_table["region_context_score"] = region_bonus
    score_table["repeat_context_score"] = repeat_bonus
    score_table["hairpin_context_score"] = hairpin_bonus
    candidates: dict[str, pd.Series] = {"baseline_score": score_table["baseline_score"]}
    for beta in betas:
        score_table[f"rbp_trace_v2_score_region__beta_{beta}"] = baseline_score_z + beta * region_bonus
        score_table[f"rbp_trace_v2_score_repeat__beta_{beta}"] = baseline_score_z + beta * repeat_bonus
        score_table[f"rbp_trace_v2_score_hairpin__beta_{beta}"] = baseline_score_z + beta * hairpin_bonus
        for beta_repeat in betas:
            for beta_hairpin in betas:
                name = f"rbp_trace_v2_score_all_context__beta_region_{beta}__beta_repeat_{beta_repeat}__beta_hairpin_{beta_hairpin}"
                score_table[name] = baseline_score_z + beta * region_bonus + beta_repeat * repeat_bonus + beta_hairpin * hairpin_bonus
    for col in score_table.columns:
        if col != "gene_id":
            candidates[col] = score_table[col]
    return candidates, score_table


def run_context_v2_pipeline(
    config: TraceConfig,
    enable_region: bool = False,
    enable_repeat: bool = False,
    enable_hairpin: bool = False,
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    out_dir = ensure_dir(output_dir or (config.results_root / "osdrb1_context_v2"))
    baseline = build_baseline_score_table(read_table(ensure_path_exists(config.paths["gene_score_table"], "gene score table")))
    truth_gene_ids = _load_truth_gene_ids(ensure_path_exists(config.paths["truth_gene_list"], "truth gene list"))

    feature_table, top3_window_features = build_context_feature_table(
        config=config,
        baseline=baseline,
        out_dir=out_dir,
        enable_region=enable_region,
        enable_repeat=enable_repeat,
        enable_hairpin=enable_hairpin,
    )

    candidates, score_table = _score_family_candidates(feature_table, config.context_betas, enable_region, enable_repeat, enable_hairpin)
    score_table["gene_id"] = feature_table["gene_id"].astype(str)
    score_table["baseline_rank"] = feature_table["baseline_rank"]

    comparable_truth = set(feature_table["gene_id"]) & truth_gene_ids
    splits = make_splits(comparable_truth, config.n_splits, config.calibration_fraction, config.random_seed)

    selected_rows = []
    heldout_rows = []
    baseline_heldout = evaluate_fixed_score(score_table, "baseline_score", splits, config.top_ks, label="baseline_score")
    baseline_heldout["score_family"] = "baseline_score"
    heldout_rows.append(baseline_heldout)
    selected_rows.append(pd.DataFrame({"seed": [split.seed for split in splits], "score_family": "baseline_score", "selected_candidate": "baseline_score"}))

    for split in splits:
        family_map = {
            "rbp_trace_v2_score_region": [f"rbp_trace_v2_score_region__beta_{beta}" for beta in config.context_betas],
            "rbp_trace_v2_score_repeat": [f"rbp_trace_v2_score_repeat__beta_{beta}" for beta in config.context_betas],
            "rbp_trace_v2_score_hairpin": [f"rbp_trace_v2_score_hairpin__beta_{beta}" for beta in config.context_betas],
            "rbp_trace_v2_score_all_context": [name for name in candidates if name.startswith("rbp_trace_v2_score_all_context__")],
        }
        for family, names in family_map.items():
            subset = {name: candidates[name] for name in names}
            best_name, cal_summary = choose_best_beta(score_table, subset, split, config.top_ks)
            row = {"seed": split.seed, "score_family": family, "selected_candidate": best_name}
            if family == "rbp_trace_v2_score_all_context":
                parts = best_name.split("__")
                row["beta_region"] = float(parts[1].split("_")[-1])
                row["beta_repeat"] = float(parts[2].split("_")[-1])
                row["beta_hairpin"] = float(parts[3].split("_")[-1])
            else:
                row["selected_beta"] = float(best_name.split("__beta_")[-1])
            selected_rows.append(pd.DataFrame([row]))
            held = evaluate_topk(score_table.assign(candidate_score=candidates[best_name]), "candidate_score", split.heldout_truth, config.top_ks, label=family, seed=split.seed)
            held["score_family"] = family
            held["selected_candidate"] = best_name
            heldout_rows.append(held)
            if not (out_dir / "calibration_candidate_summary.tsv").exists():
                cal_summary.to_csv(out_dir / "calibration_candidate_summary.tsv", sep="\t", index=False)
            else:
                cal_summary.to_csv(out_dir / "calibration_candidate_summary.tsv", sep="\t", index=False, mode="a", header=False)

    selected_params = pd.concat(selected_rows, ignore_index=True)
    selected_params.to_csv(out_dir / "context_v2_selected_params.tsv", sep="\t", index=False)
    heldout_summary = pd.concat(heldout_rows, ignore_index=True)
    heldout_summary.to_csv(out_dir / "context_v2_heldout_summary.tsv", sep="\t", index=False)

    family_map = {
        "rbp_trace_v2_score_region": [f"rbp_trace_v2_score_region__beta_{beta}" for beta in config.context_betas],
        "rbp_trace_v2_score_repeat": [f"rbp_trace_v2_score_repeat__beta_{beta}" for beta in config.context_betas],
        "rbp_trace_v2_score_hairpin": [f"rbp_trace_v2_score_hairpin__beta_{beta}" for beta in config.context_betas],
        "rbp_trace_v2_score_all_context": [name for name in candidates if name.startswith("rbp_trace_v2_score_all_context__")],
    }
    full_truth_rows = [evaluate_topk(score_table.assign(score=score_table["baseline_score"]), "score", truth_gene_ids, config.top_ks, label="baseline_score_oracle").assign(score_family="baseline_score")]
    for family, names in family_map.items():
        counts = Counter(selected_params[selected_params["score_family"] == family]["selected_candidate"].tolist())
        if counts:
            chosen = counts.most_common(1)[0][0]
            score_table[family] = candidates[chosen]
            full_truth_rows.append(evaluate_topk(score_table.assign(score=candidates[chosen]), "score", truth_gene_ids, config.top_ks, label=f"{family}_oracle").assign(score_family=family, selected_candidate=chosen))
    full_truth = pd.concat(full_truth_rows, ignore_index=True)
    full_truth.to_csv(out_dir / "context_v2_full_truth_diagnostic.tsv", sep="\t", index=False)
    score_table.to_csv(out_dir / "context_v2_score_table.tsv", sep="\t", index=False)

    audit_col = "baseline_score"
    counts = Counter(selected_params[selected_params["score_family"] == "rbp_trace_v2_score_all_context"]["selected_candidate"].tolist())
    if counts:
        chosen = counts.most_common(1)[0][0]
        audit_col = chosen
        score_table["context_enhanced_score"] = candidates[chosen]
    else:
        score_table["context_enhanced_score"] = score_table["baseline_score"]
    audit_frame = feature_table.merge(score_table[["gene_id", "context_enhanced_score"]], on="gene_id", how="left")
    audit_top200 = assign_tp_fp_groups(audit_frame.rename(columns={"context_enhanced_score": "score"}), "score", truth_gene_ids, 200)
    audit_top1000 = assign_tp_fp_groups(audit_frame.rename(columns={"context_enhanced_score": "score"}), "score", truth_gene_ids, 1000)
    audit_all = pd.concat([audit_top200.assign(top_k_group=200), audit_top1000.assign(top_k_group=1000)], ignore_index=True)
    audit_all.to_csv(out_dir / "context_v2_tp_fp_audit.tsv", sep="\t", index=False)
    summary_cols = [
        "baseline_top3_mean_score",
        "top3_CDS_fraction",
        "top3_UTR_fraction",
        "top3_inverted_repeat_score_mean",
        "top3_palindrome_score_mean",
        "top3_reverse_complement_4mer_density_mean",
        "top3_reverse_complement_5mer_density_mean",
        "top3_reverse_complement_6mer_density_mean",
        "top3_stem_like_score_mean",
        "max_longest_high_paired_run",
        "top3_paired_run_density_mean",
        "n_windows",
    ]
    summarize_groups(audit_all, [col for col in summary_cols if col in audit_all.columns]).to_csv(out_dir / "context_v2_tp_fp_feature_summary.tsv", sep="\t", index=False)

    baseline_top200 = baseline_heldout[baseline_heldout["top_k"] == 200]["overlap"].mean()
    context_top200 = heldout_summary[(heldout_summary["score_family"] == "rbp_trace_v2_score_all_context") & (heldout_summary["top_k"] == 200)]["overlap"].mean()
    report = [
        "# RBP_TRACE_V2 CONTEXT REPORT",
        "",
        f"- baseline held-out Top200 mean overlap: {baseline_top200:.2f}",
        f"- baseline held-out Top500 mean overlap: {baseline_heldout[baseline_heldout['top_k'] == 500]['overlap'].mean():.2f}",
        f"- baseline held-out Top1000 mean overlap: {baseline_heldout[baseline_heldout['top_k'] == 1000]['overlap'].mean():.2f}",
        f"- region module held-out Top200 mean overlap: {heldout_summary[(heldout_summary['score_family'] == 'rbp_trace_v2_score_region') & (heldout_summary['top_k'] == 200)]['overlap'].mean():.2f}",
        f"- repeat module held-out Top200 mean overlap: {heldout_summary[(heldout_summary['score_family'] == 'rbp_trace_v2_score_repeat') & (heldout_summary['top_k'] == 200)]['overlap'].mean():.2f}",
        f"- hairpin module held-out Top200 mean overlap: {heldout_summary[(heldout_summary['score_family'] == 'rbp_trace_v2_score_hairpin') & (heldout_summary['top_k'] == 200)]['overlap'].mean():.2f}",
        f"- all_context held-out Top200 mean overlap: {context_top200:.2f}",
        "",
    ]
    if not np.isfinite(context_top200) or context_top200 <= baseline_top200:
        report.append("Existing sequence/structure/context features still do not exceed baseline under held-out validation.")
    else:
        report.append("All-context score exceeds baseline under held-out validation.")
    write_lines(out_dir / "RBP_TRACE_V2_CONTEXT_REPORT.md", report)
    return {
        "out_dir": out_dir,
        "feature_table": out_dir / "rbp_trace_v2_gene_feature_table.tsv",
        "score_table": out_dir / "context_v2_score_table.tsv",
        "full_truth": out_dir / "context_v2_full_truth_diagnostic.tsv",
        "heldout": out_dir / "context_v2_heldout_summary.tsv",
        "selected_params": out_dir / "context_v2_selected_params.tsv",
        "report": out_dir / "RBP_TRACE_V2_CONTEXT_REPORT.md",
        "audit": out_dir / "context_v2_tp_fp_audit.tsv",
    }
