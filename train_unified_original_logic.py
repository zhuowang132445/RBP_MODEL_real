#!/usr/bin/env python3
"""Stage-wise training for the original-logic unified RBP model."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))
from unified_config import get_config
from unified_original_logic_model import OriginalLogicUnifiedRBPModel

UC_REPEAT_PROBES = ("CUUCUCU", "UUUCUCU", "CUCUCUU", "UCUCUCU", "UUCUCUU", "CUCUUCU")
G_RICH_PROBES = ("UGGAGUG", "UAGAACG", "AGUAGGA", "GAUUGGA", "GGAUGGA", "GAUGGAA")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clean_seq(seq):
    return str(seq).upper().replace("*", "")


def phys_feats(seq):
    seq = clean_seq(seq)
    n = max(len(seq), 1)
    return [len(seq) / 1000.0, seq.count("G") / n, seq.count("R") / n, seq.count("K") / n]


def kingdom_label(tax_kingdom):
    x = str(tax_kingdom).lower()
    if "viridiplantae" in x or "plant" in x:
        return 1
    if "fung" in x:
        return 2
    if "metazoa" in x or "animal" in x:
        return 0
    return 3


def build_kmer_index(kmers: np.ndarray) -> dict[str, int]:
    return {str(k).upper().replace("T", "U"): i for i, k in enumerate(kmers.astype(str).tolist())}


def derive_conflict_targets(zscores: np.ndarray, kmers: np.ndarray, uc_threshold: float, g_threshold: float):
    kmer_index = build_kmer_index(kmers)
    uc_idx = [kmer_index[k] for k in UC_REPEAT_PROBES if k in kmer_index]
    g_idx = [kmer_index[k] for k in G_RICH_PROBES if k in kmer_index]
    if not uc_idx or not g_idx:
        raise ValueError("required UC/G probe kmers missing from motif profile vocabulary")
    uc_scores = zscores[:, uc_idx].max(axis=1)
    g_scores = zscores[:, g_idx].max(axis=1)
    uc_target = (uc_scores >= float(uc_threshold)).astype(np.float32)
    g_target = (g_scores >= float(g_threshold)).astype(np.float32)
    conflict_target = ((uc_target > 0.5) & (g_target > 0.5)).astype(np.float32)
    return (
        uc_target,
        g_target,
        conflict_target,
        uc_scores.astype(np.float32),
        g_scores.astype(np.float32),
    )


class MotifDataset(Dataset):
    def __init__(self, resource_dir: Path, split: str, seed=42, uc_threshold: float = 0.5, g_threshold: float = 0.5):
        meta = pd.read_csv(resource_dir / "motif_profiles.tsv", sep="\t")
        npz = np.load(resource_dir / "motif_profiles.npz")
        ids = npz["profile_ids"].astype(str).tolist()
        z = npz["zscores"].astype(np.float32)
        self.kmers = npz["kmers"].astype(str)
        meta = meta.set_index("rnacompete_id").loc[ids].reset_index()
        rng = np.random.default_rng(seed)
        idx = np.arange(len(ids))
        rng.shuffle(idx)
        n_test = max(1, int(round(len(idx) * 0.15)))
        n_val = max(1, int(round(len(idx) * 0.15)))
        if split == "test":
            use = idx[:n_test]
        elif split == "val":
            use = idx[n_test:n_test + n_val]
        else:
            use = idx[n_test + n_val:]
        self.meta = meta.iloc[use].reset_index(drop=True)
        self.z = z[use]
        self.mask = npz["zscore_mask"].astype(np.float32)[use] if "zscore_mask" in npz.files else np.isfinite(self.z).astype(np.float32)
        self.z = np.nan_to_num(self.z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        uc_target, g_target, conflict_target, _, _ = derive_conflict_targets(
            self.z,
            self.kmers,
            uc_threshold=uc_threshold,
            g_threshold=g_threshold,
        )
        self.uc_target = uc_target
        self.g_target = g_target
        self.conflict_target = conflict_target

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, i):
        row = self.meta.iloc[i]
        seq = clean_seq(row["protein_sequence"])
        return (
            seq,
            torch.tensor(phys_feats(seq), dtype=torch.float32),
            torch.tensor(kingdom_label(row.get("tax_kingdom", "")), dtype=torch.long),
            torch.from_numpy(self.z[i]),
            torch.from_numpy(self.mask[i]),
            torch.tensor(self.uc_target[i], dtype=torch.float32),
            torch.tensor(self.g_target[i], dtype=torch.float32),
            torch.tensor(self.conflict_target[i], dtype=torch.float32),
        )


def motif_collate(batch):
    seqs = [x[0] for x in batch]
    return (
        seqs,
        torch.stack([x[1] for x in batch]),
        torch.stack([x[2] for x in batch]),
        torch.stack([x[3] for x in batch]),
        torch.stack([x[4] for x in batch]),
        torch.stack([x[5] for x in batch]),
        torch.stack([x[6] for x in batch]),
        torch.stack([x[7] for x in batch]),
    )


class ExternalRescueMotifDataset(Dataset):
    def __init__(
        self,
        profile_csv: Path,
        fasta_path: Path,
        kmers: np.ndarray,
        rbp_id_to_query_protein_id: dict[str, str],
        rbp_ids: list[str] | None = None,
        uc_threshold: float = 0.5,
        g_threshold: float = 0.5,
    ):
        frame = pd.read_csv(profile_csv, sep=None, engine="python")
        if "kmer" not in frame.columns:
            frame = frame.rename(columns={frame.columns[0]: "kmer"})
        frame["kmer"] = frame["kmer"].astype(str).str.upper().str.replace("T", "U", regex=False)
        frame = frame.drop_duplicates(subset=["kmer"], keep="first").set_index("kmer")
        aligned = frame.reindex([str(k).upper().replace("T", "U") for k in kmers.tolist()])
        aligned = aligned.fillna(0.0)
        fasta_records = {}
        header = None
        parts = []
        with open(fasta_path, "rt", encoding="utf-8", errors="ignore") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if header is not None:
                        fasta_records[header] = "".join(parts)
                    header = line[1:].split()[0]
                    parts = []
                else:
                    parts.append(line)
        if header is not None:
            fasta_records[header] = "".join(parts)

        use_ids = rbp_ids or [x for x in rbp_id_to_query_protein_id if x in aligned.columns]
        rows = []
        targets = []
        for rbp_id in use_ids:
            query_id = rbp_id_to_query_protein_id[rbp_id]
            seq = fasta_records.get(query_id)
            if not seq:
                raise ValueError(f"external rescue fasta missing sequence for {query_id}")
            candidate_col = rbp_id if rbp_id in aligned.columns else query_id
            if candidate_col not in aligned.columns:
                raise ValueError(f"external rescue profile missing column for {rbp_id}/{query_id}")
            rows.append(
                {
                    "rbp_id": rbp_id,
                    "query_id": query_id,
                    "protein_sequence": clean_seq(seq),
                    "tax_kingdom": "Viridiplantae",
                }
            )
            targets.append(aligned[candidate_col].to_numpy(dtype=np.float32))
        self.meta = pd.DataFrame(rows)
        self.z = np.stack(targets).astype(np.float32)
        self.mask = np.ones_like(self.z, dtype=np.float32)
        uc_target, g_target, conflict_target, _, _ = derive_conflict_targets(
            self.z,
            kmers,
            uc_threshold=uc_threshold,
            g_threshold=g_threshold,
        )
        self.uc_target = uc_target
        self.g_target = g_target
        self.conflict_target = conflict_target

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, i):
        row = self.meta.iloc[i]
        seq = clean_seq(row["protein_sequence"])
        return (
            seq,
            torch.tensor(phys_feats(seq), dtype=torch.float32),
            torch.tensor(kingdom_label(row.get("tax_kingdom", "")), dtype=torch.long),
            torch.from_numpy(self.z[i]),
            torch.from_numpy(self.mask[i]),
            torch.tensor(self.uc_target[i], dtype=torch.float32),
            torch.tensor(self.g_target[i], dtype=torch.float32),
            torch.tensor(self.conflict_target[i], dtype=torch.float32),
        )


class BindingDataset(Dataset):
    def __init__(self, resource_dir: Path, split: str, motif_cache_dir: Path):
        report = json.load(open(resource_dir / "multitask_data_report.json"))
        cache_dir = Path(report["sequence_cache_dir"])
        self.rna = np.load(cache_dir / f"{split}.rna_codes.npy", mmap_mode="r")
        self.protein_rows = np.load(cache_dir / f"{split}.protein_rows.npy", mmap_mode="r")
        self.labels = np.load(cache_dir / f"{split}.labels.npy", mmap_mode="r")
        idx = pd.read_csv(motif_cache_dir / "motif_feature_index.tsv", sep="\t")
        train_idx = idx[idx.source == "music_clip_train"].sort_values("embedding_row")
        latent = np.load(motif_cache_dir / "motif_latent.npy")
        subtype = np.load(motif_cache_dir / "motif_subtype_logits.npy")
        seed = np.load(motif_cache_dir / "motif_seed_logits.npy")
        row_positions = train_idx.index.to_numpy()
        self.motif = np.concatenate([latent[row_positions], subtype[row_positions], seed[row_positions]], axis=1).astype(np.float32)
        mean = self.motif.mean(axis=0, keepdims=True)
        std = self.motif.std(axis=0, keepdims=True)
        std[std < 1e-6] = 1.0
        self.motif = ((self.motif - mean) / std).astype(np.float32)
        self.motif = np.nan_to_num(self.motif, nan=0.0, posinf=10.0, neginf=-10.0).astype(np.float32)
        if self.motif.shape[0] <= int(np.max(self.protein_rows)):
            raise ValueError("motif cache does not cover all protein rows")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        prow = int(self.protein_rows[i])
        return (
            torch.from_numpy(np.asarray(self.rna[i], dtype=np.int64)),
            torch.tensor(prow, dtype=torch.long),
            torch.from_numpy(self.motif[prow]),
            torch.tensor(float(self.labels[i]), dtype=torch.float32),
        )


def binary_eval(labels, probs):
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs)
    if len(np.unique(labels)) < 2:
        return {"auc": float("nan"), "auprc": float("nan")}
    return {"auc": float(roc_auc_score(labels, probs)), "auprc": float(average_precision_score(labels, probs))}


@torch.no_grad()
def eval_binding(model, loader, device, max_steps=200):
    model.eval()
    labels, probs = [], []
    for step, batch in enumerate(loader, 1):
        if max_steps and step > max_steps:
            break
        rna, prow, motif, y = [x.to(device) for x in batch]
        logit = model.forward_binding_motif_aware(rna, prow, motif)
        labels.append(y.cpu().numpy())
        probs.append(torch.sigmoid(logit).cpu().numpy())
    labels = np.concatenate(labels)
    probs = np.concatenate(probs)
    finite = np.isfinite(probs)
    if not finite.all():
        raise RuntimeError(f"non-finite binding probabilities during evaluation: {int((~finite).sum())}/{len(probs)}")
    return binary_eval(labels, probs)


@torch.no_grad()
def eval_motif(model, loader, device):
    model.eval()
    losses, raw_losses, cors = [], [], []
    for seqs, pf, kl, z, mask, uc_t, g_t, conflict_t in loader:
        pf, kl, z, mask = pf.to(device), kl.to(device), z.to(device), mask.to(device)
        out = model.forward_motif(seqs, pf, kl)
        pred = out["rescued_z"]
        raw_pred = out["reconstructed_z"]
        losses.append(float((nn.functional.smooth_l1_loss(pred, z, reduction="none") * mask).sum().div(mask.sum().clamp_min(1.0)).item()))
        raw_losses.append(float((nn.functional.smooth_l1_loss(raw_pred, z, reduction="none") * mask).sum().div(mask.sum().clamp_min(1.0)).item()))
        a, b = pred.cpu().numpy(), z.cpu().numpy()
        for x, y in zip(a, b):
            if np.std(x) > 0 and np.std(y) > 0:
                cors.append(float(np.corrcoef(x, y)[0, 1]))
    return {
        "motif_loss": float(np.mean(losses)),
        "motif_raw_loss": float(np.mean(raw_losses)),
        "motif_pearson": float(np.mean(cors)) if cors else float("nan"),
    }


def set_stage_trainability(model, stage):
    for p in model.parameters():
        p.requires_grad = False
    if stage in {"motif", "joint"}:
        for name, p in model.motif_model.named_parameters():
            if not name.startswith("esm2."):
                p.requires_grad = True
        for p in model.motif_group_classifier.parameters():
            p.requires_grad = True
        for p in model.motif_conflict_classifier.parameters():
            p.requires_grad = True
        for p in model.motif_rescue_latent_head.parameters():
            p.requires_grad = True
        model.motif_group_latent_prototypes.requires_grad = True
    if stage in {"binding", "joint", "late_fusion"}:
        for p in model.binding_model.parameters():
            p.requires_grad = True
        for p in model.motif_binding_projector.parameters():
            p.requires_grad = True
        for p in model.motif_aware_classifier.parameters():
            p.requires_grad = True


def masked_smooth_l1(pred, target, mask):
    per_element = nn.functional.smooth_l1_loss(pred, target, reduction="none")
    masked = per_element * mask
    return masked.sum().div(mask.sum().clamp_min(1.0))


def compute_motif_loss_terms(model, seqs, pf, kl, z, mask, uc_t, g_t, conflict_t, device):
    pf = pf.to(device)
    kl = kl.to(device)
    z = z.to(device)
    mask = mask.to(device)
    uc_t = uc_t.to(device)
    g_t = g_t.to(device)
    conflict_t = conflict_t.to(device)
    out = model.forward_motif(seqs, pf, kl)
    group_targets = torch.stack([uc_t, g_t], dim=1)
    return {
        "output": out,
        "raw_loss": masked_smooth_l1(out["reconstructed_z"], z, mask),
        "rescue_loss": masked_smooth_l1(out["rescued_z"], z, mask),
        "group_loss": nn.functional.binary_cross_entropy_with_logits(out["motif_group_logits"], group_targets),
        "conflict_loss": nn.functional.binary_cross_entropy_with_logits(out["motif_conflict_logit"], conflict_t),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["motif", "binding", "joint", "late_fusion"], default="late_fusion")
    ap.add_argument("--resource-dir", default="data/multitask")
    ap.add_argument("--motif-cache-dir", default="data/original_logic_cache")
    ap.add_argument("--out-dir", default="results/original_logic_training")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--motif-steps-per-epoch", type=int, default=30)
    ap.add_argument("--binding-steps-per-epoch", type=int, default=500)
    ap.add_argument("--batch-size-motif", type=int, default=4)
    ap.add_argument("--batch-size-binding", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rescue-loss-weight", type=float, default=1.0)
    ap.add_argument("--raw-motif-loss-weight", type=float, default=0.2)
    ap.add_argument("--motif-group-loss-weight", type=float, default=0.05)
    ap.add_argument("--motif-conflict-loss-weight", type=float, default=0.05)
    ap.add_argument("--uc-threshold", type=float, default=0.5)
    ap.add_argument("--g-threshold", type=float, default=0.5)
    ap.add_argument("--external-rescue-profile-csv", default=None)
    ap.add_argument("--external-rescue-fasta", default=None)
    ap.add_argument("--external-rescue-rbp-ids", default="w1,w2,w3,w4,w5,w6")
    ap.add_argument("--external-rescue-loss-weight", type=float, default=0.5)
    ap.add_argument("--batch-size-external-rescue", type=int, default=6)
    ap.add_argument("--dry-run-check", action="store_true")
    args = ap.parse_args()

    cfg = get_config()
    set_seed(cfg.seed)
    res, cache, out = Path(args.resource_dir), Path(args.motif_cache_dir), Path(args.out_dir)
    ckpt = Path(args.checkpoint or f"checkpoints/unified_original_logic_model_{args.stage}.pt")
    if not res.is_absolute(): res = Path(cfg.work_dir) / res
    if not cache.is_absolute(): cache = Path(cfg.work_dir) / cache
    if not out.is_absolute(): out = Path(cfg.work_dir) / out
    if not ckpt.is_absolute(): ckpt = Path(cfg.work_dir) / ckpt
    external_rescue_profile_csv = Path(args.external_rescue_profile_csv) if args.external_rescue_profile_csv else None
    if external_rescue_profile_csv is not None and not external_rescue_profile_csv.is_absolute():
        external_rescue_profile_csv = Path(cfg.work_dir) / external_rescue_profile_csv
    external_rescue_fasta = Path(args.external_rescue_fasta) if args.external_rescue_fasta else None
    if external_rescue_fasta is not None and not external_rescue_fasta.is_absolute():
        external_rescue_fasta = Path(cfg.work_dir) / external_rescue_fasta
    out.mkdir(parents=True, exist_ok=True)
    ckpt.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = OriginalLogicUnifiedRBPModel(cfg, load_pretrained=True).to(device)
    if args.resume:
        resume = Path(args.resume)
        if not resume.is_absolute(): resume = Path(cfg.work_dir) / resume
        state = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state_dict"], strict=False)
    set_stage_trainability(model, args.stage)

    motif_train_ds = MotifDataset(res, "train", uc_threshold=args.uc_threshold, g_threshold=args.g_threshold)
    motif_val_ds = MotifDataset(res, "val", uc_threshold=args.uc_threshold, g_threshold=args.g_threshold)
    motif_train = DataLoader(motif_train_ds, batch_size=args.batch_size_motif, shuffle=True, collate_fn=motif_collate)
    motif_val = DataLoader(motif_val_ds, batch_size=args.batch_size_motif, shuffle=False, collate_fn=motif_collate)
    bind_train = DataLoader(BindingDataset(res, "train", cache), batch_size=args.batch_size_binding, shuffle=True)
    bind_val = DataLoader(BindingDataset(res, "val", cache), batch_size=args.batch_size_binding, shuffle=False)
    ext_rescue_train = None
    if external_rescue_profile_csv is not None and external_rescue_fasta is not None:
        ext_ids = [x.strip() for x in str(args.external_rescue_rbp_ids).split(",") if x.strip()]
        ext_ds = ExternalRescueMotifDataset(
            profile_csv=external_rescue_profile_csv,
            fasta_path=external_rescue_fasta,
            kmers=motif_train_ds.kmers,
            rbp_id_to_query_protein_id=cfg.rbp_id_to_query_protein_id,
            rbp_ids=ext_ids,
            uc_threshold=args.uc_threshold,
            g_threshold=args.g_threshold,
        )
        ext_rescue_train = DataLoader(
            ext_ds,
            batch_size=min(int(args.batch_size_external_rescue), max(1, len(ext_ds))),
            shuffle=True,
            collate_fn=motif_collate,
        )

    if args.dry_run_check:
        seqs, pf, kl, z, mask, uc_t, g_t, conflict_t = next(iter(motif_train))
        rna, prow, motif, y = next(iter(bind_train))
        with torch.no_grad():
            motif_out = model.forward_motif(seqs, pf.to(device), kl.to(device))
            motif_pred = motif_out["rescued_z"]
            bind_pred = model.forward_binding_motif_aware(rna.to(device), prow.to(device), motif.to(device))
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        report = {
            "status": "passed",
            "stage": args.stage,
            "trainable_parameters": int(trainable),
            "motif_shape": list(motif_pred.shape),
            "binding_shape": list(bind_pred.shape),
            "motif_group_shape": list(motif_out["motif_group_logits"].shape),
            "device": str(device),
        }
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return

    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError(f"no trainable parameters for stage={args.stage}")
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()
    hist = []
    for ep in range(1, args.epochs + 1):
        model.train()
        mit, bit, rows = iter(motif_train), iter(bind_train), []
        exit_it = iter(ext_rescue_train) if ext_rescue_train is not None else None
        n_steps = {
            "motif": args.motif_steps_per_epoch,
            "binding": args.binding_steps_per_epoch,
            "joint": max(args.motif_steps_per_epoch, args.binding_steps_per_epoch),
            "late_fusion": args.binding_steps_per_epoch,
        }[args.stage]
        for step in range(n_steps):
            opt.zero_grad()
            loss = torch.zeros((), device=device)
            log = {}
            if args.stage in {"motif", "joint"} and step < args.motif_steps_per_epoch:
                try:
                    seqs, pf, kl, z, mask, uc_t, g_t, conflict_t = next(mit)
                except StopIteration:
                    mit = iter(motif_train)
                    seqs, pf, kl, z, mask, uc_t, g_t, conflict_t = next(mit)
                motif_terms = compute_motif_loss_terms(
                    model,
                    seqs,
                    pf,
                    kl,
                    z,
                    mask,
                    uc_t,
                    g_t,
                    conflict_t,
                    device,
                )
                motif_total = (
                    float(args.rescue_loss_weight) * motif_terms["rescue_loss"]
                    + float(args.raw_motif_loss_weight) * motif_terms["raw_loss"]
                    + float(args.motif_group_loss_weight) * motif_terms["group_loss"]
                    + float(args.motif_conflict_loss_weight) * motif_terms["conflict_loss"]
                )
                loss = loss + motif_total
                log["motif_loss"] = float(motif_total.item())
                log["motif_rescue_loss"] = float(motif_terms["rescue_loss"].item())
                log["motif_raw_loss"] = float(motif_terms["raw_loss"].item())
                log["motif_group_loss"] = float(motif_terms["group_loss"].item())
                log["motif_conflict_loss"] = float(motif_terms["conflict_loss"].item())
                if exit_it is not None and args.external_rescue_loss_weight > 0:
                    try:
                        ext_batch = next(exit_it)
                    except StopIteration:
                        exit_it = iter(ext_rescue_train)
                        ext_batch = next(exit_it)
                    ext_seqs, ext_pf, ext_kl, ext_z, ext_mask, ext_uc_t, ext_g_t, ext_conflict_t = ext_batch
                    ext_terms = compute_motif_loss_terms(
                        model,
                        ext_seqs,
                        ext_pf,
                        ext_kl,
                        ext_z,
                        ext_mask,
                        ext_uc_t,
                        ext_g_t,
                        ext_conflict_t,
                        device,
                    )
                    ext_total = float(args.external_rescue_loss_weight) * (
                        float(args.rescue_loss_weight) * ext_terms["rescue_loss"]
                        + float(args.motif_group_loss_weight) * ext_terms["group_loss"]
                        + float(args.motif_conflict_loss_weight) * ext_terms["conflict_loss"]
                    )
                    loss = loss + ext_total
                    log["external_rescue_loss"] = float(ext_total.item())
                    log["external_rescue_recon_loss"] = float(ext_terms["rescue_loss"].item())
                    log["external_group_loss"] = float(ext_terms["group_loss"].item())
                    log["external_conflict_loss"] = float(ext_terms["conflict_loss"].item())
            if args.stage in {"binding", "joint", "late_fusion"} and step < args.binding_steps_per_epoch:
                try:
                    rna, prow, motif, y = next(bit)
                except StopIteration:
                    bit = iter(bind_train)
                    rna, prow, motif, y = next(bit)
                pred = model.forward_binding_motif_aware(rna.to(device), prow.to(device), motif.to(device))
                bl = bce(pred, y.to(device))
                loss = loss + bl
                log["binding_loss"] = float(bl.item())
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite training loss at epoch={ep} step={step}: {float(loss.detach().cpu())}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 5.0)
            opt.step()
            rows.append(log)
        metrics = {}
        if args.stage in {"motif", "joint"}:
            metrics.update(eval_motif(model, motif_val, device))
        if args.stage in {"binding", "joint", "late_fusion"}:
            metrics.update(eval_binding(model, bind_val, device))
        row = {"epoch": ep, "stage": args.stage, **{k: float(np.mean([r[k] for r in rows if k in r])) for k in {k for r in rows for k in r}}, **metrics}
        hist.append(row)
        print(json.dumps(row, ensure_ascii=False))

    pd.DataFrame(hist).to_csv(out / f"original_logic_{args.stage}_training_history.tsv", sep="\t", index=False)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": cfg.to_dict(),
            "args": vars(args),
            "stage": args.stage,
            "binding_score_mode": "joint" if args.stage == "late_fusion" else "base",
            "use_joint_head": bool(args.stage == "late_fusion"),
        },
        ckpt,
    )
    report = {"status": "completed", "stage": args.stage, "checkpoint": str(ckpt), "last_metrics": hist[-1] if hist else {}}
    with open(out / f"original_logic_{args.stage}_train_report.json", "w") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
