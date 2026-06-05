from __future__ import annotations

from typing import Iterable

import numpy as np


def safe_mean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    return float(np.nanmean(arr)) if np.isfinite(arr).any() else float("nan")


def safe_max(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    return float(np.nanmax(arr)) if np.isfinite(arr).any() else float("nan")
