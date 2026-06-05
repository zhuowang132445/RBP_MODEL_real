#!/usr/bin/env python3
"""Build auditable resources for unified RNAcompete + MuSIC CLIP multitask training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from unified_config import get_config

AA_ALLOWED = set("ACDEFGHIKLMNPQRSTVWYXBZUO")


def read_fasta(path: Path) -> dict[str, str]:
    records = {}
    header = None
    parts = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records[header] = "".join(parts)
                header = line[1:]
                parts = []
            else:
                parts.append("".join(c for c in line.upper() if c.isalpha() or c == "*"))
    if header is not None:
        records[header] = "".join(parts)
    return records


def clean_protein(seq: str) -> str:
    seq = str(seq).upper().replace("*", "")
    return "".join(c if c in AA_ALLOWED else "X" for c in seq)


def parse_clip_fasta_lookup(fasta_path: Path) -> dict[tuple[str, str, str], str]:
    lookup = {}
    for header, seq in read_fasta(fasta_path).items():
        first = header.split()[0]
        fields = first.split("|")
        if len(fields) < 4:
            continue
        rbp, species = fields[0], fields[1]
        resolved = None
        requested = None
        for field in fields[2:]:
            if field.startswith("resolved="):
                resolved = field.split("=", 1)[1]
            if field.startswith("requested="):
                requested = field.split("=", 1)[1]
        for acc in {resolved, requested}:
            if acc:
                lookup[(rbp, species, acc)] = clean_protein(seq)
    return lookup


def build_motif_resources(cfg, out_dir: Path) -> dict:
    motif_dir = Path(cfg.motif_snapshot_dir) / "data"
    z = pd.read_csv(motif_dir / "zscore_train.tsv", sep="\t")
    meta = pd.read_csv(motif_dir / "rnacompete_metadata_eupri.tsv", sep="\t")
    profile_ids = [c for c in z.columns if c != "kmer"]
    meta = meta[meta["rnacompete_id"].isin(profile_ids)].copy()
    meta = meta.drop_duplicates("rnacompete_id", keep="first")
    meta = meta.set_index("rnacompete_id").loc[profile_ids].reset_index()
    seq_col = "construct_aa_seq" if "construct_aa_seq" in meta.columns else "protein_aa_seq"
    meta["protein_sequence"] = meta[seq_col].map(clean_protein)
    meta["protein_length"] = meta["protein_sequence"].str.len()
    keep = meta["protein_length"] >= 30
    meta = meta.loc[keep].reset_index(drop=True)
    profile_ids = meta["rnacompete_id"].tolist()
    y_raw = z[profile_ids].to_numpy(dtype=np.float32).T
    y_mask = np.isfinite(y_raw).astype(np.float32)
    y = np.nan_to_num(y_raw, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    kmers = z["kmer"].astype(str).to_numpy()
    np.savez_compressed(out_dir / "motif_profiles.npz", profile_ids=np.array(profile_ids, dtype=str), kmers=np.array(kmers, dtype=str), zscores=y, zscore_mask=y_mask)
    cols = ["rnacompete_id", "gene_name", "protein_id", "tax_name", "protein_sequence", "protein_length"]
    meta[[c for c in cols if c in meta.columns]].to_csv(out_dir / "motif_profiles.tsv", sep="\t", index=False)
    return {"n_motif_profiles": int(len(profile_ids)), "n_kmers": int(len(kmers)), "motif_profile_ids": profile_ids[:5]}


def build_clip_resources(cfg, out_dir: Path) -> dict:
    clip_root = Path(cfg.binding_project_dir)
    idx = pd.read_csv(clip_root / "08_embeddings/esm2_t33_650M_strict_fixed/protein_embedding_index.tsv", sep="\t")
    fasta_lookup = parse_clip_fasta_lookup(next(x for x in [clip_root / "04_rbp_protein/MuSIC_RBP_proteins_all.strict_repaired.fasta", clip_root / "04_rbp_protein/MuSIC_RBP_proteins_all.fly_human_mouse_repaired.fasta", clip_root / "04_rbp_protein/MuSIC_RBP_proteins_all.fasta"] if x.exists()))
    seqs = []
    missing = []
    for _, row in idx.iterrows():
        key = (str(row["rbp_name"]), str(row["species_abbrev"]), str(row["protein_acc"]))
        seq = fasta_lookup.get(key)
        if not seq:
            missing.append(str(row["protein_id"]))
            seq = ""
        seqs.append(seq)
    idx = idx.copy()
    idx["protein_sequence"] = seqs
    idx["sequence_found"] = idx["protein_sequence"].str.len() > 0
    idx.to_csv(out_dir / "clip_protein_sequences.tsv", sep="\t", index=False)
    split_dir = clip_root / "07_training_data/window_dataset_top1000_strict_fixed/split_by_experiment_accession"
    cache_dir = clip_root / "09_models/rbp_binding_cnn_esm2_t33_650M_strict_fixed_experiment_holdout/cache"
    struct_dir = clip_root / "07_training_data/window_dataset_top1000_strict_fixed/structure_features/rnafold_pf_paired_probability_split_by_experiment_accession"
    return {
        "n_clip_proteins": int(len(idx)),
        "n_clip_proteins_with_sequence": int(idx["sequence_found"].sum()),
        "missing_sequence_count": int(len(missing)),
        "missing_sequence_examples": missing[:10],
        "split_dir": str(split_dir),
        "sequence_cache_dir": str(cache_dir),
        "structure_cache_dir": str(struct_dir),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/multitask")
    args = parser.parse_args()
    cfg = get_config()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = Path(cfg.work_dir) / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"status": "completed", "out_dir": str(out_dir)}
    report.update(build_motif_resources(cfg, out_dir))
    report.update(build_clip_resources(cfg, out_dir))
    with open(out_dir / "multitask_data_report.json", "w") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
