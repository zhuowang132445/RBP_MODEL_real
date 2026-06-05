from __future__ import annotations

from typing import Dict

import numpy as np


def run_lengths(mask: np.ndarray) -> list[int]:
    runs: list[int] = []
    current = 0
    for flag in mask.astype(bool).tolist():
        if flag:
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return runs


def max_sliding_mean(values: np.ndarray, width: int) -> float:
    if values.size == 0:
        return float("nan")
    if values.size < width:
        return float(np.nanmean(values))
    kernel = np.ones(width, dtype=float) / float(width)
    return float(np.nanmax(np.convolve(values.astype(float), kernel, mode="valid")))


def compute_hairpin_features(paired: np.ndarray) -> Dict[str, float]:
    paired = np.asarray(paired, dtype=float)
    paired = paired[np.isfinite(paired)]
    if paired.size == 0:
        return {
            "longest_high_paired_run": float("nan"),
            "high_paired_run_count": float("nan"),
            "paired_run_density": float("nan"),
            "paired_unpaired_boundary_count": float("nan"),
            "max_local_paired_mean_31nt": float("nan"),
            "max_local_paired_mean_41nt": float("nan"),
            "max_local_paired_mean_81nt": float("nan"),
            "stem_like_score": float("nan"),
            "structure_peakiness_score": float("nan"),
            "MFE": float("nan"),
            "MFE_per_nt": float("nan"),
            "normalized_MFE": float("nan"),
            "predicted_stem_count": float("nan"),
            "predicted_longest_stem_length": float("nan"),
            "predicted_hairpin_loop_size": float("nan"),
            "predicted_stem_loop_score": float("nan"),
        }
    high = paired >= 0.7
    runs = run_lengths(high)
    boundaries = int(np.sum(high[1:] != high[:-1])) if paired.size > 1 else 0
    local31 = max_sliding_mean(paired, 31)
    local41 = max_sliding_mean(paired, 41)
    local81 = max_sliding_mean(paired, 81)
    global_mean = float(np.nanmean(paired))
    stem_like = float(np.nanmean([min((max(runs) if runs else 0) / 40.0, 1.0), local41, float(np.mean(high))]))
    peakiness = float(max(local31, local41, local81) - global_mean)
    return {
        "longest_high_paired_run": float(max(runs) if runs else 0),
        "high_paired_run_count": float(len(runs)),
        "paired_run_density": float(np.mean(high)),
        "paired_unpaired_boundary_count": float(boundaries),
        "max_local_paired_mean_31nt": local31,
        "max_local_paired_mean_41nt": local41,
        "max_local_paired_mean_81nt": local81,
        "stem_like_score": stem_like,
        "structure_peakiness_score": peakiness,
        "MFE": float("nan"),
        "MFE_per_nt": float("nan"),
        "normalized_MFE": float("nan"),
        "predicted_stem_count": float("nan"),
        "predicted_longest_stem_length": float("nan"),
        "predicted_hairpin_loop_size": float("nan"),
        "predicted_stem_loop_score": float("nan"),
    }
