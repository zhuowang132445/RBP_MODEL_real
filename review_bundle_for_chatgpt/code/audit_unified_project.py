#!/usr/bin/env python3
"""Audit unified_rbp_model_v1 project files, data links, and ID consistency."""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))
from train_unified_stage1 import load_rbp_features, read_table
from unified_config import get_config
from unified_model import UnifiedRBPModel, stage1_window_feature_columns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-model", action="store_true", help="Instantiate full frozen backbones. This may load large ESM weights.")
    args = parser.parse_args()
    cfg = get_config()
    report = {"status": "passed", "errors": [], "warnings": []}

    required_paths = {
        "motif_checkpoint": cfg.motif_checkpoint,
        "motif_model_py": cfg.motif_model_py,
        "binding_checkpoint": cfg.binding_checkpoint,
        "binding_train_py": cfg.binding_train_py,
        "binding_train_embedding_npy": cfg.binding_train_embedding_npy,
        "stage1_window_features": cfg.stage1_window_features,
        "stage1_rbp_features_json": cfg.stage1_rbp_features_json,
    }
    for name, path in required_paths.items():
        if not os.path.exists(path):
            report["status"] = "failed"
            report["errors"].append(f"{name} not found: {path}")

    if report["status"] == "passed":
        df = read_table(cfg.stage1_window_features)
        rbp = load_rbp_features(cfg.stage1_rbp_features_json)
        feature_cols = stage1_window_feature_columns(cfg)
        missing_cols = sorted(set(feature_cols + ["label", "rbp_fasta_id", "gene_id", "transcript_id"]) - set(df.columns))
        if missing_cols:
            report["status"] = "failed"
            report["errors"].append(f"stage1 feature table missing columns: {missing_cols}")
        missing_rbp = sorted(set(df["rbp_fasta_id"].astype(str)) - set(rbp["protein_ids"]))
        if missing_rbp:
            report["status"] = "failed"
            report["errors"].append(f"rbp_fasta_id missing from rbp_features: {missing_rbp[:10]}")
        report["stage1_rows"] = int(len(df))
        report["stage1_rbps"] = df["rbp_id"].value_counts().to_dict()
        report["feature_dim"] = int(len(feature_cols))
        report["rbp_feature_count"] = int(len(rbp["protein_ids"]))
        if "base_binding_logit" in df.columns:
            zero_frac = df.assign(_z=df["base_binding_logit"].fillna(0).astype(float).eq(0)).groupby("rbp_id")["_z"].mean().to_dict()
            report["base_binding_logit_zero_fraction_by_rbp"] = {str(k): float(v) for k, v in zero_frac.items()}
            bad = {str(k): float(v) for k, v in zero_frac.items() if v > 0.95}
            if bad:
                report["warnings"].append(f"base_binding_logit nearly all zero for: {bad}")

    if args.load_model and report["status"] == "passed":
        try:
            model = UnifiedRBPModel(cfg, load_backbones=True)
            report["full_model_load"] = "passed"
            del model
        except Exception as exc:
            report["status"] = "failed"
            report["errors"].append(f"full model load failed: {exc}")

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "unified_project_audit.json", "w") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["status"] != "passed":
        sys.exit(1)


if __name__ == "__main__":
    main()
