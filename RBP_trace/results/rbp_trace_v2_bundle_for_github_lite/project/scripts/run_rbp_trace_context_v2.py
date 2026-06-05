#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rbp_trace.config import load_config
from rbp_trace.pipelines.run_baseline import run_baseline_pipeline
from rbp_trace.pipelines.run_context_v2 import run_context_v2_pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--enable-region", action="store_true")
    parser.add_argument("--enable-repeat", action="store_true")
    parser.add_argument("--enable-hairpin", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if not any([args.enable_region, args.enable_repeat, args.enable_hairpin]):
        run_baseline_pipeline(config)
        return
    run_context_v2_pipeline(
        config=config,
        enable_region=args.enable_region,
        enable_repeat=args.enable_repeat,
        enable_hairpin=args.enable_hairpin,
    )


if __name__ == "__main__":
    main()
