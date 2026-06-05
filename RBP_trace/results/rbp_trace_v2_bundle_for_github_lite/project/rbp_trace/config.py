from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TraceConfig:
    raw: dict[str, Any]

    @property
    def results_root(self) -> Path:
        return Path(self.raw["results_root"])

    @property
    def paths(self) -> dict[str, str]:
        return dict(self.raw.get("paths", {}))

    @property
    def baseline(self) -> dict[str, Any]:
        return dict(self.raw.get("baseline", {}))

    @property
    def top_ks(self) -> list[int]:
        return [int(x) for x in self.raw.get("top_ks", [50, 100, 200, 500, 1000])]

    @property
    def n_splits(self) -> int:
        return int(self.raw.get("n_splits", 10))

    @property
    def random_seed(self) -> int:
        return int(self.raw.get("random_seed", 20260605))

    @property
    def calibration_fraction(self) -> float:
        return float(self.raw.get("calibration_fraction", 0.7))

    @property
    def context_betas(self) -> list[float]:
        return [float(x) for x in self.raw.get("context_betas", [0.1, 0.25, 0.5, 1.0])]


def load_config(path: str | Path) -> TraceConfig:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return TraceConfig(raw=data)
