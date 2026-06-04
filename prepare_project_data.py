#!/usr/bin/env python3
"""Create in-folder data/source links and manifest for unified_rbp_model_v1."""

import argparse
import json
import os
from pathlib import Path

from unified_config import get_config


def safe_symlink(src: str, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    os.symlink(src, dst)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--copy-large-files", action="store_true", help="Reserved; current implementation uses symlinks.")
    args = parser.parse_args()
    cfg = get_config()
    work = Path(cfg.work_dir)
    data = Path(cfg.data_dir)
    sources = Path(cfg.data_sources_dir)
    results = Path(cfg.output_dir)
    ckpts = Path(cfg.checkpoint_dir)
    for d in [work, data, sources, results, ckpts, work / "scripts"]:
        d.mkdir(parents=True, exist_ok=True)

    links = {
        sources / "motif_model_snapshot": cfg.motif_snapshot_dir,
        sources / "binding_project": cfg.binding_project_dir,
        data / "motif_train" / "seq_train.fasta": f"{cfg.motif_snapshot_dir}/data/seq_train.fasta",
        data / "motif_train" / "zscore_train.tsv": f"{cfg.motif_snapshot_dir}/data/zscore_train.tsv",
        data / "motif_train" / "rnacompete_metadata_eupri.tsv": f"{cfg.motif_snapshot_dir}/data/rnacompete_metadata_eupri.tsv",
        data / "binding_train" / "experiment_holdout_split": f"{cfg.binding_project_dir}/07_training_data/window_dataset_top1000_strict_fixed/split_by_experiment_accession",
        data / "binding_train" / "rnafold_structure_cache": f"{cfg.binding_project_dir}/07_training_data/window_dataset_top1000_strict_fixed/structure_features/rnafold_pf_paired_probability_split_by_experiment_accession",
        data / "binding_train" / "protein_embeddings.npy": cfg.binding_train_embedding_npy,
        data / "binding_train" / "protein_embedding_index.tsv": cfg.binding_train_embedding_index,
        data / "stage1_validation" / "fusion_v2_window_features.tsv.gz": cfg.stage1_window_features,
        data / "stage1_validation" / "rbp_features.json": cfg.stage1_rbp_features_json,
        data / "checkpoints" / "motif_v6_1.pth": cfg.motif_checkpoint,
        data / "checkpoints" / "binding_cnn_best_model.pt": cfg.binding_checkpoint,
    }

    manifest_rows = []
    for dst, src in links.items():
        safe_symlink(src, dst)
        p = Path(src)
        manifest_rows.append({
            "name": str(dst.relative_to(work)),
            "path_in_project": str(dst),
            "source_path": src,
            "exists": bool(p.exists()),
            "is_symlink": True,
        })

    manifest = {
        "project": "unified_rbp_model_v1",
        "work_dir": cfg.work_dir,
        "storage_policy": "large original/intermediate files are included as symlinks inside this folder",
        "links": manifest_rows,
    }
    with open(work / "DATA_MANIFEST.json", "w") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    with open(work / "DATA_MANIFEST.tsv", "w") as handle:
        handle.write("name\tpath_in_project\tsource_path\texists\tis_symlink\n")
        for row in manifest_rows:
            handle.write("\t".join(str(row[k]) for k in ["name", "path_in_project", "source_path", "exists", "is_symlink"]) + "\n")
    print(json.dumps({"status": "completed", "manifest": str(work / "DATA_MANIFEST.json")}, indent=2))


if __name__ == "__main__":
    main()
