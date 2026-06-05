#!/usr/bin/env python3
"""Cache V6.1 motif features for MuSIC CLIP proteins and external validation proteins."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))
from unified_config import get_config
from unified_original_logic_model import OriginalLogicUnifiedRBPModel

AA = set("ACDEFGHIKLMNPQRSTVWYXBZUO")


def read_fasta(path: Path) -> dict[str, str]:
    rec = {}
    h = None
    parts = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if h is not None:
                    rec[h] = "".join(parts)
                h = line[1:]
                parts = []
            else:
                parts.append("".join(c if c in AA else "X" for c in line.upper().replace("*", "")))
    if h is not None:
        rec[h] = "".join(parts)
    return rec


def phys_feats(seq: str):
    seq = str(seq).upper()
    n = max(len(seq), 1)
    return [len(seq) / 1000.0, seq.count("G") / n, seq.count("R") / n, seq.count("K") / n]


def kingdom_label(species_abbrev: str) -> int:
    plant = {"ARATH", "ORYSJ", "ORYSA", "MAIZE", "SOLLC"}
    return 1 if str(species_abbrev).upper() in plant else 0


def parse_clip_fasta(path: Path) -> dict[tuple[str, str, str], str]:
    out = {}
    for h, s in read_fasta(path).items():
        first = h.split()[0]
        fields = first.split("|")
        if len(fields) < 4:
            continue
        rbp, species = fields[0], fields[1]
        vals = []
        for f in fields[2:]:
            if f.startswith("resolved=") or f.startswith("requested="):
                vals.append(f.split("=", 1)[1])
        for acc in vals:
            out[(rbp, species, acc)] = s
    return out


def load_sequences(cfg):
    clip_root = Path(cfg.binding_project_dir)
    idx = pd.read_csv(cfg.binding_train_embedding_index, sep="\t").sort_values("embedding_row")
    fasta = next(x for x in [
        clip_root / "04_rbp_protein/MuSIC_RBP_proteins_all.strict_repaired.fasta",
        clip_root / "04_rbp_protein/MuSIC_RBP_proteins_all.fly_human_mouse_repaired.fasta",
        clip_root / "04_rbp_protein/MuSIC_RBP_proteins_all.fasta",
    ] if x.exists())
    lookup = parse_clip_fasta(fasta)
    rows = []
    for _, r in idx.iterrows():
        key = (str(r.rbp_name), str(r.species_abbrev), str(r.protein_acc))
        seq = lookup.get(key, "")
        rows.append({"source": "music_clip_train", "protein_id": r.protein_id, "embedding_row": int(r.embedding_row), "species_abbrev": r.species_abbrev, "sequence": seq})
    ext_fasta = Path(cfg.binding_project_dir) / "08_embeddings/plant_rbp_extended_query_fasta/rice7_plus_validated_plant_rbp.fasta"
    if ext_fasta.exists():
        for h, s in read_fasta(ext_fasta).items():
            if "|GRP7|" in h:
                rows.append({"source": "external", "protein_id": "AtGRP7", "embedding_row": -1, "species_abbrev": "ARATH", "sequence": s})
            elif "|GRP8|" in h:
                rows.append({"source": "external", "protein_id": "AtGRP8", "embedding_row": -1, "species_abbrev": "ARATH", "sequence": s})
            elif "LOC_Os05g24160.1" in h:
                rows.append({"source": "external", "protein_id": "LOC_Os05g24160.1", "embedding_row": -1, "species_abbrev": "ORYSJ", "sequence": s})
    df = pd.DataFrame(rows)
    missing = df[df.sequence.str.len() == 0].protein_id.tolist()
    if missing:
        raise ValueError(f"missing protein sequences for motif cache: {missing[:20]}")
    return df


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/original_logic_cache")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    cfg = get_config()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = Path(cfg.work_dir) / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_sequences(cfg)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = OriginalLogicUnifiedRBPModel(cfg, load_pretrained=True).to(device)
    model.eval()
    rows = []
    latents = []
    subtypes = []
    seeds = []
    for i in range(0, len(df), args.batch_size):
        batch = df.iloc[i:i + args.batch_size]
        seqs = batch.sequence.tolist()
        pf = torch.tensor([phys_feats(s) for s in seqs], dtype=torch.float32, device=device)
        kl = torch.tensor([kingdom_label(x) for x in batch.species_abbrev], dtype=torch.long, device=device)
        out = model.forward_motif(seqs, pf, kl)
        latents.append(out["motif_latent"].detach().cpu().numpy().astype(np.float32))
        subtypes.append(out["motif_subtype_logits"].detach().cpu().numpy().astype(np.float32))
        seeds.append(out["motif_seed_logits"].detach().cpu().numpy().astype(np.float32))
        rows.extend(batch[["source", "protein_id", "embedding_row", "species_abbrev"]].to_dict("records"))
        print(f"[CACHE] processed {min(i + args.batch_size, len(df))}/{len(df)}")
    np.save(out_dir / "motif_latent.npy", np.concatenate(latents, axis=0))
    np.save(out_dir / "motif_subtype_logits.npy", np.concatenate(subtypes, axis=0))
    np.save(out_dir / "motif_seed_logits.npy", np.concatenate(seeds, axis=0))
    pd.DataFrame(rows).to_csv(out_dir / "motif_feature_index.tsv", sep="\t", index=False)
    report = {"status": "completed", "n_proteins": int(len(rows)), "out_dir": str(out_dir)}
    with open(out_dir / "motif_feature_cache_report.json", "w") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
