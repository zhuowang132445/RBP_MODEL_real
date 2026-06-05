from __future__ import annotations

import pandas as pd

from rbp_trace.evaluation.overlap_eval import evaluate_topk
from rbp_trace.scoring.baseline_scores import build_baseline_score_table


def test_baseline_score_table_can_read_in_memory() -> None:
    frame = pd.DataFrame(
        {
            "gene_id": ["g1", "g2", "g3"],
            "transcript_id": ["g1.1", "g2.1", "g3.1"],
            "top3_mean_score": [0.9, 0.2, 0.8],
            "n_windows": [3, 2, 4],
            "max_score": [1.0, 0.3, 0.85],
        }
    )
    out = build_baseline_score_table(frame)
    assert out.iloc[0]["gene_id"] == "g1"
    assert "binding_potential_score" in out.columns


def test_topk_overlap_eval_runs() -> None:
    frame = pd.DataFrame({"gene_id": ["g1", "g2", "g3"], "score": [0.9, 0.1, 0.8]})
    eval_df = evaluate_topk(frame, "score", {"g1", "g3"}, [1, 2, 3], label="toy")
    assert list(eval_df["top_k"]) == [1, 2, 3]
    assert int(eval_df[eval_df["top_k"] == 2]["overlap"].iloc[0]) == 2
