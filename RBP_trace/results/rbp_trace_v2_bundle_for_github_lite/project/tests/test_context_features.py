from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rbp_trace.config import load_config
from rbp_trace.modules.hairpin_stem_loop import compute_hairpin_features
from rbp_trace.modules.region_annotation import annotate_region_features
from rbp_trace.modules.repeat_inverted_repeat import compute_repeat_features
from rbp_trace.pipelines.run_context_v2 import run_context_v2_pipeline


def test_region_annotation_missing_gtf_outputs_na(tmp_path: Path) -> None:
    frame = pd.DataFrame({"gene_id": ["g1"], "transcript_id": ["tx1"], "window_start": [1], "window_end": [20], "window_rank": [1]})
    out = annotate_region_features(frame, None, tmp_path / "region.tsv", tmp_path / "missing.txt")
    assert out.iloc[0]["region_type"] == "NA"
    assert (tmp_path / "missing.txt").exists()


def test_repeat_sequence_only_features_compute() -> None:
    feat = compute_repeat_features("AUGCAUGCAUGCAUGC")
    assert "inverted_repeat_score" in feat
    assert feat["reverse_complement_4mer_density"] >= 0


def test_hairpin_paired_features_compute() -> None:
    vec = np.array([0.1, 0.8, 0.9, 0.85, 0.2, 0.1, 0.75, 0.8], dtype=float)
    feat = compute_hairpin_features(vec)
    assert feat["longest_high_paired_run"] >= 2
    assert feat["stem_like_score"] >= 0


def test_context_pipeline_completes_without_optional_annotation(tmp_path: Path) -> None:
    project = tmp_path
    gene = pd.DataFrame(
        {
            "gene_id": ["g1", "g2", "g3", "g4"],
            "transcript_id": ["g1.1", "g2.1", "g3.1", "g4.1"],
            "n_windows": [2, 2, 2, 2],
            "best_window_start": [1, 1, 1, 1],
            "best_window_end": [20, 20, 20, 20],
            "best_window_seq": ["AUGCAUGCAUGCAUGCAUGC"] * 4,
            "max_score": [0.95, 0.8, 0.7, 0.2],
            "top3_mean_score": [0.9, 0.7, 0.6, 0.1],
        }
    )
    gene_path = project / "gene.tsv"
    gene.to_csv(gene_path, sep="\t", index=False)

    windows = pd.DataFrame(
        {
            "gene_id": ["g1", "g1", "g2", "g2", "g3", "g3", "g4", "g4"],
            "transcript_id": ["g1.1", "g1.1", "g2.1", "g2.1", "g3.1", "g3.1", "g4.1", "g4.1"],
            "window_start": [1, 5, 1, 5, 1, 5, 1, 5],
            "window_end": [20, 24, 20, 24, 20, 24, 20, 24],
            "rna_seq": ["AUGCAUGCAUGCAUGCAUGC"] * 8,
            "inverted_base_motif_structure_score": [0.95, 0.85, 0.8, 0.7, 0.65, 0.6, 0.2, 0.1],
            "paired_probability_mean": [0.6, 0.7, 0.6, 0.65, 0.55, 0.6, 0.2, 0.1],
            "fraction_high_paired": [0.4, 0.5, 0.35, 0.4, 0.3, 0.35, 0.1, 0.05],
            "base_score": [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.7, 0.75],
        }
    )
    window_path = project / "windows.tsv.gz"
    windows.to_csv(window_path, sep="\t", index=False, compression="gzip")
    window_table = windows[["transcript_id", "window_start", "window_end"]].copy()
    window_table_path = project / "window_table.tsv.gz"
    window_table.to_csv(window_table_path, sep="\t", index=False, compression="gzip")
    np.save(project / "paired.npy", np.tile(np.linspace(0.1, 0.9, 24), (len(window_table), 1)))
    truth = pd.DataFrame({"gene_id": ["g1", "g2"]})
    truth_path = project / "truth.tsv"
    truth.to_csv(truth_path, sep="\t", index=False)

    cfg_path = project / "cfg.yaml"
    cfg_path.write_text(
        f"""
project_name: RBP-TRACE
results_root: {project / "results"}
random_seed: 1
n_splits: 2
calibration_fraction: 0.5
top_ks: [1, 2, 200, 500]
context_betas: [0.1, 0.5]
baseline:
  score_name: inverted_base_motif_structure_score
  aggregation: top3_mean_score
paths:
  gene_score_table: {gene_path}
  window_score_table: {window_path}
  truth_gene_list: {truth_path}
  window_table: {window_table_path}
  structure_npy: {project / "paired.npy"}
  annotation_gtf: ""
  repeat_annotation: ""
  transcript_gene_map: ""
""",
        encoding="utf-8",
    )
    config = load_config(cfg_path)
    result = run_context_v2_pipeline(config, enable_region=True, enable_repeat=True, enable_hairpin=True)
    assert result["feature_table"].exists()
    assert result["heldout"].exists()
