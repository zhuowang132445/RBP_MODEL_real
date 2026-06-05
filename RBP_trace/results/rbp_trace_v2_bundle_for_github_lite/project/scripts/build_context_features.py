#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rbp_trace.config import load_config
from rbp_trace.pipelines.run_context_v2 import build_context_feature_table
from rbp_trace.io_utils import read_table
from rbp_trace.scoring.baseline_scores import build_baseline_score_table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--enable-region", action="store_true")
    parser.add_argument("--enable-repeat", action="store_true")
    parser.add_argument("--enable-hairpin", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    baseline = build_baseline_score_table(read_table(config.paths["gene_score_table"]))
    out_dir = config.results_root / "osdrb1_context_v2"
    build_context_feature_table(
        config=config,
        baseline=baseline,
        out_dir=out_dir,
        enable_region=args.enable_region,
        enable_repeat=args.enable_repeat,
        enable_hairpin=args.enable_hairpin,
    )


if __name__ == "__main__":
    main()
