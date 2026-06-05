from __future__ import annotations

import numpy as np
import pandas as pd


def robust_zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    median = values.median(skipna=True)
    mad = (values - median).abs().median(skipna=True)
    if pd.isna(mad) or mad == 0:
        std = values.std(skipna=True)
        if pd.isna(std) or std == 0:
            return pd.Series(np.zeros(len(values)), index=series.index, dtype=float)
        return (values - values.mean(skipna=True)) / std
    return 0.67448975 * (values - median) / mad
