from __future__ import annotations

from pathlib import Path

import pandas as pd

from rbp_trace.config import TraceConfig
from rbp_trace.evaluation.heldout_validation import evaluate_fixed_score, make_splits
from rbp_trace.evaluation.overlap_eval import evaluate_topk
from rbp_trace.evaluation.tp_fp_audit import assign_tp_fp_groups, summarize_groups
from rbp_trace.io_utils import ensure_dir, ensure_path_exists, read_table, write_lines
from rbp_trace.scoring.baseline_scores import build_baseline_score_table


def _load_truth_gene_ids(path: str | Path) -> set[str]:
    frame = read_table(path)
    if "gene_id" not in frame.columns:
        raise ValueError(f"truth gene list missing gene_id column: {path}")
    return {str(x) for x in frame["gene_id"].dropna().astype(str).tolist()}


def run_baseline_pipeline(config: TraceConfig, output_dir: str | Path | None = None) -> dict[str, Path]:
    out_dir = ensure_dir(output_dir or (config.results_root / "osdrb1_baseline"))
    gene_score_path = ensure_path_exists(config.paths["gene_score_table"], "gene score table")
    truth_path = ensure_path_exists(config.paths["truth_gene_list"], "truth gene list")

    gene_scores = read_table(gene_score_path)
    truth_gene_ids = _load_truth_gene_ids(truth_path)
    baseline = build_baseline_score_table(
        gene_scores=gene_scores,
        score_name=config.baseline.get("score_name", "inverted_base_motif_structure_score"),
        aggregation=config.baseline.get("aggregation", "top3_mean_score"),
    )
    baseline_path = out_dir / "gene_scores_baseline.tsv"
    baseline.to_csv(baseline_path, sep="\t", index=False)

    full_truth = evaluate_topk(baseline, "baseline_score", truth_gene_ids, config.top_ks, label="baseline_full_truth")
    overlap_path = out_dir / "baseline_overlap_summary.tsv"
    full_truth.to_csv(overlap_path, sep="\t", index=False)

    comparable_truth = set(baseline["gene_id"]) & truth_gene_ids
    splits = make_splits(comparable_truth, config.n_splits, config.calibration_fraction, config.random_seed)
    heldout = evaluate_fixed_score(baseline, "baseline_score", splits, config.top_ks, label="baseline_score")
    heldout_path = out_dir / "baseline_heldout_summary.tsv"
    heldout.to_csv(heldout_path, sep="\t", index=False)

    audit_top200 = assign_tp_fp_groups(baseline, "baseline_score", truth_gene_ids, top_k=200)
    audit_top1000 = assign_tp_fp_groups(baseline, "baseline_score", truth_gene_ids, top_k=1000)
    audit = pd.concat([audit_top200.assign(top_k_group=200), audit_top1000.assign(top_k_group=1000)], ignore_index=True)
    audit_summary = summarize_groups(audit, ["baseline_score", "n_windows", "max_score", "top3_mean_score"])
    audit_path = out_dir / "baseline_tp_fp_audit.tsv"
    audit_summary.to_csv(audit_path, sep="\t", index=False)

    full_top200 = full_truth[full_truth["top_k"] == 200].iloc[0]
    held_top200 = heldout[heldout["top_k"] == 200]
    report_lines = [
        "# BASELINE REPORT",
        "",
        f"- baseline score source: {config.baseline.get('score_name', 'inverted_base_motif_structure_score')} aggregated by {config.baseline.get('aggregation', 'top3_mean_score')}",
        f"- full truth Top200 overlap: {int(full_top200['overlap'])}",
        f"- held-out Top200 mean overlap: {held_top200['overlap'].mean():.2f}",
        f"- held-out Top500 mean overlap: {heldout[heldout['top_k'] == 500]['overlap'].mean():.2f}",
        f"- held-out Top1000 mean overlap: {heldout[heldout['top_k'] == 1000]['overlap'].mean():.2f}",
        "",
        "Full-truth metrics are oracle diagnostics. Held-out metrics are the main generalization estimate.",
    ]
    report_path = out_dir / "BASELINE_REPORT.md"
    write_lines(report_path, report_lines)
    return {
        "out_dir": out_dir,
        "baseline_table": baseline_path,
        "full_truth": overlap_path,
        "heldout": heldout_path,
        "audit": audit_path,
        "report": report_path,
    }
