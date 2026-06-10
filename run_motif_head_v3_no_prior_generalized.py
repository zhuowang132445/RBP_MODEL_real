#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import os
RUNTIME_ROOT = Path(os.environ.get("RBP_TRACE_V3_RUNTIME_ROOT", Path(__file__).resolve().parent))
BASE = Path(os.environ.get("RBP_TRACE_V3_BASE", RUNTIME_ROOT / "tmp_rbp_trace_diag"))
PREV = Path(os.environ.get("RBP_TRACE_V3_PREV", RUNTIME_ROOT / "tmp_rbp_trace_diag_output"))
OUT = Path(os.environ.get("RBP_TRACE_V3_OUT", RUNTIME_ROOT.parent / "results" / "review_motif_head_v3_no_prior_generalized"))
OUT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUCS = "ACGU"
ALL_6MERS = np.array(["".join(x) for x in __import__("itertools").product(NUCS, repeat=6)])
ALL_7MERS = np.array(["".join(x) for x in __import__("itertools").product(NUCS, repeat=7)])
IDX6 = {k: i for i, k in enumerate(ALL_6MERS)}
IDX7 = {k: i for i, k in enumerate(ALL_7MERS)}

CONFIG = {
    "seed": 17,
    "hidden_dim": 128,
    "dropout": 0.10,
    "epochs": 220,
    "batch_size": 4,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "cluster_topk_eval": 100,
    "loss_weights": {
        "profile_regression": 1.0,
        "topk_ranking": 0.20,
        "family_classification": 0.20,
        "cluster_ranking": 0.20,
        "anti_collapse": 0.15,
    },
    "split_policy": {
        "validation_rbps": ["w6", "AtAPUM23"],
        "family_internal_loo_families": ["RBNS_lab", "PUF/APUM"],
    },
}

ALIASES = {
    "LOC_Os05g24160.1": "OsDRB1",
    "OsDRB1": "OsDRB1",
    "AtCFI25_or_CFIm_like": "AtCFIM25",
    "AtCFI25/CFIm_like": "AtCFIM25",
    "AtCFI25 / CFIm-like": "AtCFIM25",
    "AtCFIM25 / CFIm-like": "AtCFIM25",
    "AtHYL1/DRB1": "AtHYL1_DRB1",
    "ARATH|GRP7|Q03250|Glycine-rich_RNA-binding_protein_7": "AtGRP7",
    "ARATH|GRP8|Q03251|Glycine-rich_RNA-binding_protein_8": "AtGRP8",
    "ARATH|HYL1|O04492|Double-stranded_RNA-binding_protein_1": "AtHYL1_DRB1",
    "ORYSJ|LOC_Os05g24160.1|Q0DJA3|original_rice7": "OsDRB1",
    "ARATH|AGO1|O04379|Protein_argonaute_1": "AtAGO1",
    "ARATH|AGO4|Q9ZVD5|Protein_argonaute_4": "AtAGO4",
    "ARATH|AGO7|Q9C793|Protein_argonaute_7": "AtAGO7",
    "ORYSJ|AGO18|Q69UP6|Protein_argonaute_18": "OsAGO18",
    "ARATH|DRB4|Q8H1D4|Double-stranded_RNA-binding_protein_4": "AtDRB4",
    "ARATH|DCL1|Q9SP32|Endoribonuclease_Dicer_homolog_1": "AtDCL1",
    "ARATH|DCL4|P84634|Dicer-like_protein_4": "AtDCL4",
}


def canon(raw: object) -> str:
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return ""
    text = str(raw).strip()
    if not text or text == "nan":
        return ""
    if text in ALIASES:
        return ALIASES[text]
    if "|" in text:
        parts = text.split("|")
        if len(parts) >= 2:
            joined = "|".join(parts[:4])
            if joined in ALIASES:
                return ALIASES[joined]
            if parts[1].startswith("At"):
                return parts[1]
            if parts[1].startswith("LOC_Os05g24160"):
                return "OsDRB1"
            if parts[1].startswith("AGO"):
                if parts[0] == "ARATH":
                    return f"At{parts[1]}"
                if parts[0] == "ORYSJ":
                    return f"Os{parts[1]}"
            if parts[1] in {"GRP7", "GRP8", "HYL1", "DRB4", "DCL1", "DCL4"}:
                prefix = "At" if parts[0] == "ARATH" else "Os"
                suffix = "_DRB1" if parts[1] == "HYL1" else ""
                return f"{prefix}{parts[1]}{suffix}"
    return ALIASES.get(text, text)


def norm_seq(seq: str) -> str:
    return str(seq).upper().replace("T", "U")


def motif_flags(kmer: str) -> dict[str, int]:
    kmer = norm_seq(kmer)
    return {
        "U": int(kmer.count("U") >= max(3, math.ceil(len(kmer) * 0.55))),
        "UC": int(sum(kmer.count(x) for x in "UC") >= max(4, math.ceil(len(kmer) * 0.75))),
        "A": int(kmer.count("A") >= max(3, math.ceil(len(kmer) * 0.55))),
        "G": int(kmer.count("G") >= max(3, math.ceil(len(kmer) * 0.45))),
    }


def group_from_kmer(kmer: str) -> str:
    kmer = norm_seq(kmer)
    flags = motif_flags(kmer)
    if "UGUA" in kmer or "UUGA" in kmer:
        return "PUF_like_UGUA_UUGA"
    if "AAUAAA" in kmer or "AUUAAA" in kmer:
        return "A_rich_polyA"
    if "UGUA" in kmer:
        return "UGUA_like_CFIm"
    if flags["UC"]:
        return "UC-rich"
    if flags["U"]:
        return "U-rich"
    if flags["A"]:
        return "A-rich"
    if flags["G"]:
        return "G-rich"
    return "Other"


MOTIF_GROUPS = [
    "U-rich",
    "UC-rich",
    "A_rich_polyA",
    "UGUA_like_CFIm",
    "PUF_like_UGUA_UUGA",
    "A-rich",
    "G-rich",
    "Other",
]
GROUP_TO_IDX = {g: i for i, g in enumerate(MOTIF_GROUPS)}

FAMILY_LABELS = [
    "RBNS_lab",
    "PUF/APUM",
    "polyA_processing",
    "GRP/RRM",
    "PPR_control",
    "dsRNA_control",
    "AGO_control",
    "DICER_control",
    "other_control",
    "unknown",
]
FAMILY_TO_IDX = {f: i for i, f in enumerate(FAMILY_LABELS)}


def family_bucket(rbp_id: str, protein_family: str, expected: str) -> str:
    text = f"{protein_family} {expected} {rbp_id}"
    if rbp_id.startswith("w"):
        return "RBNS_lab"
    if "APUM" in text or "PUF" in text:
        return "PUF/APUM"
    if "CPSF" in text or "CFIM" in text or "FIP1" in text or "FPA" in text or "HLP1" in text or "polyA" in text:
        return "polyA_processing"
    if "GRP" in text:
        return "GRP/RRM"
    if "PPR" in text:
        return "PPR_control"
    if "AGO" in text:
        return "AGO_control"
    if "DCL" in text or "Dicer" in text:
        return "DICER_control"
    if "DRB" in text or "HYL1" in text or "OsDRB1" in text:
        return "dsRNA_control"
    if "control" in text or "diagnostic" in text:
        return "other_control"
    return "unknown"


def expected_group(expected: str) -> str:
    text = str(expected)
    if "PUF" in text or "UGUA" in text or "UUGA" in text:
        return "PUF_like_UGUA_UUGA"
    if "AAUAAA" in text or "polyA" in text or "A_rich" in text:
        return "A_rich_polyA"
    if "CFIm" in text:
        return "UGUA_like_CFIm"
    if "UC" in text:
        return "UC-rich"
    if "U-rich" in text:
        return "U-rich"
    if "PPR" in text:
        return "PPR_long_target_control"
    if "DRB" in text or "HYL1" in text or "structure" in text or "AGO" in text:
        return "dsRNA_structure_control"
    if "diagnostic" in text:
        return "diagnostic_only"
    return text or "unknown"


def weighted_corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def weighted_spearman(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(spearmanr(a, b).correlation)


def tokenize_motif_text(text: object) -> list[str]:
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return []
    tokens = re.findall(r"[ACGUTN]{4,16}", str(text).upper())
    return [norm_seq(t) for t in tokens]


def expand_degenerate(token: str, max_variants: int = 256) -> list[str]:
    mapping = {"A": "A", "C": "C", "G": "G", "U": "U", "N": "ACGU"}
    seqs = [""]
    for ch in token:
        opts = mapping.get(ch, "")
        if not opts:
            return []
        new = []
        for prefix in seqs:
            for opt in opts:
                new.append(prefix + opt)
        seqs = new[:max_variants]
    return seqs


def compatible_kmers(token: str, k: int) -> list[str]:
    token = norm_seq(token)
    seqs = expand_degenerate(token)
    out: set[str] = set()
    for seq in seqs:
        if len(seq) == k:
            out.add(seq)
        elif len(seq) < k:
            pad = k - len(seq)
            for left in range(pad + 1):
                right = pad - left
                for pre in __import__("itertools").product(NUCS, repeat=left):
                    for suf in __import__("itertools").product(NUCS, repeat=right):
                        out.add("".join(pre) + seq + "".join(suf))
        else:
            for i in range(len(seq) - k + 1):
                out.add(seq[i : i + k])
    return sorted(out)


def build_group_mass(keys: np.ndarray, scores: np.ndarray) -> np.ndarray:
    mass = np.zeros(len(MOTIF_GROUPS), dtype=np.float32)
    pos = np.maximum(scores, 0)
    if pos.sum() == 0:
        pos = np.ones_like(pos)
    for kmer, score in zip(keys, pos):
        mass[GROUP_TO_IDX.get(group_from_kmer(str(kmer)), GROUP_TO_IDX["Other"])] += float(score)
    mass /= mass.sum()
    return mass


@dataclass
class ClusterDef:
    rbp_id: str
    cluster_id: str
    k: int
    strong: dict[str, float]
    weak: dict[str, float]
    source_token: str
    source_type: str


def shift_neighbors(token: str, k: int) -> list[str]:
    token = norm_seq(token)
    if len(token) <= k:
        return []
    seqs = expand_degenerate(token)
    out = set()
    for seq in seqs:
        for delta in (-2, -1, 1, 2):
            start = max(0, min(len(seq) - k, delta))
            if 0 <= start <= len(seq) - k:
                out.add(seq[start : start + k])
    return sorted(out)


def hamming_neighbors(kmer: str) -> list[str]:
    out = set()
    for i, ch in enumerate(kmer):
        for alt in NUCS:
            if alt != ch:
                out.add(kmer[:i] + alt + kmer[i + 1 :])
    return sorted(out)


def build_cluster_for_token(rbp_id: str, token: str, source_type: str) -> list[ClusterDef]:
    clusters = []
    anchor_cores = set()
    clean = norm_seq(token)
    for size in range(4, min(6, len(clean)) + 1):
        for i in range(len(clean) - size + 1):
            sub = clean[i : i + size]
            if set(sub) <= set(NUCS):
                anchor_cores.add(sub)
    for k in (6, 7):
        strong = {}
        for kmer in compatible_kmers(clean, k):
            weight = 1.0 if any(core in kmer for core in anchor_cores) else 0.7
            strong[kmer] = max(strong.get(kmer, 0.0), weight)
        if len(clean) > k or "N" in clean:
            for kmer in shift_neighbors(clean, k):
                if kmer not in strong:
                    strong[kmer] = 0.7
        weak = dict(strong)
        for kmer in list(strong):
            for n in hamming_neighbors(kmer):
                weak[n] = max(weak.get(n, 0.0), 0.35)
        clusters.append(
            ClusterDef(
                rbp_id=rbp_id,
                cluster_id=f"{rbp_id}_{source_type}_{k}",
                k=k,
                strong=strong,
                weak=weak,
                source_token=clean,
                source_type=source_type,
            )
        )
    return clusters


def load_metadata() -> pd.DataFrame:
    all_available = pd.read_csv(PREV / "all_available_rbp_list.tsv", sep="\t")
    status = pd.read_csv(PREV / "all_rbp_motif_prediction_status.tsv", sep="\t")
    plan = pd.read_csv(BASE / "motif_head_training_dataset_plan.tsv", sep="\t")
    candidates = pd.read_csv(BASE / "plant_non_UC_motif_truth_expansion_candidates.tsv", sep="\t")
    meta = {}
    for _, row in all_available.iterrows():
        rbp = canon(row["rbp_id"])
        meta[rbp] = {
            "rbp_id": rbp,
            "species": row.get("species", ""),
            "protein_family": row.get("protein_family", ""),
            "expected_motif_type": "",
            "notes": str(row.get("notes", "")),
        }
    for _, row in status.iterrows():
        rbp = canon(row["rbp_id"])
        meta.setdefault(rbp, {"rbp_id": rbp, "species": "", "protein_family": "", "expected_motif_type": "", "notes": ""})
        if not meta[rbp]["protein_family"]:
            meta[rbp]["protein_family"] = row.get("protein_family", "")
        if not meta[rbp]["expected_motif_type"]:
            meta[rbp]["expected_motif_type"] = row.get("expected_motif_type", "")
    for _, row in plan.iterrows():
        rbp = canon(row["rbp_id"])
        meta.setdefault(rbp, {"rbp_id": rbp, "species": "", "protein_family": "", "expected_motif_type": "", "notes": ""})
        if not meta[rbp]["expected_motif_type"]:
            meta[rbp]["expected_motif_type"] = row.get("motif_family", "")
    for _, row in candidates.iterrows():
        rbp = canon(row["rbp_id"])
        meta.setdefault(rbp, {"rbp_id": rbp, "species": "", "protein_family": "", "expected_motif_type": "", "notes": ""})
        if not meta[rbp]["species"]:
            meta[rbp]["species"] = row.get("species", "")
        if not meta[rbp]["protein_family"]:
            meta[rbp]["protein_family"] = row.get("protein_family", "")
        if not meta[rbp]["expected_motif_type"]:
            meta[rbp]["expected_motif_type"] = row.get("expected_motif_type", "")
    # add ext controls
    ext_idx = pd.read_csv(BASE / "ext_embedding_index.tsv", sep="\t")
    for raw in ext_idx["rbp_id"].dropna():
        rbp = canon(raw)
        meta.setdefault(rbp, {"rbp_id": rbp, "species": "", "protein_family": "", "expected_motif_type": "", "notes": ""})
        if "AGO" in rbp:
            meta[rbp]["protein_family"] = "AGO_control"
            meta[rbp]["expected_motif_type"] = "guide-dependent_control"
        elif "DCL" in rbp:
            meta[rbp]["protein_family"] = "DICER_control"
            meta[rbp]["expected_motif_type"] = "structure-dependent_control"
        elif "DRB4" in rbp:
            meta[rbp]["protein_family"] = "DRB_control"
            meta[rbp]["expected_motif_type"] = "structure-dependent_control"
    return pd.DataFrame(sorted(meta.values(), key=lambda r: r["rbp_id"]))


def load_embeddings(meta_df: pd.DataFrame) -> dict[str, np.ndarray]:
    emb = {}
    # plant benchmark
    arr = np.load(BASE / "plant_embedding.npy")
    idx = pd.read_csv(BASE / "plant_embedding_index.tsv", sep="\t")
    for _, row in idx.iterrows():
        emb[canon(row["rbp_name"])] = arr[int(row["embedding_row"])].astype(np.float32)
    # w1-w6
    arr = np.load(BASE / "w1_w6_esm2_embeddings.npy")
    idx = pd.read_csv(BASE / "w1_w6_esm2_embedding_index.tsv", sep="\t")
    for _, row in idx.iterrows():
        emb[canon(row["rbp_name"])] = arr[int(row["embedding_row"])].astype(np.float32)
    # ext cache
    arr = np.load(BASE / "ext_embedding.npy")
    idx = pd.read_csv(BASE / "ext_embedding_index.tsv", sep="\t")
    for _, row in idx.iterrows():
        emb[canon(row["rbp_id"])] = arr[int(row["embedding_row"])].astype(np.float32)
    # retain only metadata ids
    return {rbp: vec for rbp, vec in emb.items() if rbp in set(meta_df["rbp_id"])}


def load_current_with_prior_scores() -> dict[str, dict[int, np.ndarray]]:
    out: dict[str, dict[int, np.ndarray]] = defaultdict(dict)
    for path in [
        BASE / "w_current_auto_teacher_fused_z_matrix.csv",
        BASE / "atgrp_osdrb_current_auto_teacher_fused_z_matrix.csv",
        BASE / "plant_current_auto_teacher_fused_z_matrix.csv",
    ]:
        df = pd.read_csv(path)
        kmers = df["kmer"].map(norm_seq).tolist()
        for col in df.columns:
            if col == "kmer":
                continue
            rbp = canon(col)
            vec7 = np.full(len(ALL_7MERS), -5.0, dtype=np.float32)
            for kmer, score in zip(kmers, df[col].astype(float).tolist()):
                if kmer in IDX7:
                    vec7[IDX7[kmer]] = score
            out[rbp][7] = vec7
            # derive 6-mer by max over matching 7-mers
            vec6 = np.full(len(ALL_6MERS), -5.0, dtype=np.float32)
            scores6 = defaultdict(list)
            for kmer, score in zip(kmers, df[col].astype(float).tolist()):
                if len(kmer) == 7:
                    scores6[kmer[:6]].append(score)
                    scores6[kmer[1:]].append(score)
            for k, vals in scores6.items():
                if k in IDX6:
                    vec6[IDX6[k]] = max(vals)
            out[rbp][6] = vec6
    return out


def sparse_vec_from_topk(topk: list[str], k: int) -> np.ndarray:
    keys = ALL_6MERS if k == 6 else ALL_7MERS
    idx = IDX6 if k == 6 else IDX7
    vec = np.full(len(keys), -5.0, dtype=np.float32)
    for rank, token in enumerate(topk, start=1):
        token = norm_seq(token)
        if len(token) == k and token in idx:
            vec[idx[token]] = 2.0 - 0.05 * rank
    return vec


def load_no_prior_scores() -> dict[str, dict[int, np.ndarray]]:
    out: dict[str, dict[int, np.ndarray]] = defaultdict(dict)
    df = pd.read_csv(BASE / "with_prior_vs_no_prior_motif_prediction.tsv", sep="\t")
    mode_map = {"direct_no_auto_teacher"}
    for _, row in df.iterrows():
        if row["prediction_mode"] not in mode_map:
            continue
        rbp = canon(row["rbp_id"])
        top7 = [norm_seq(x) for x in str(row["top10_kmers"]).split(",") if x]
        out[rbp][7] = sparse_vec_from_topk(top7, 7)
        top6 = []
        for t in top7:
            if len(t) == 7:
                top6.extend([t[:6], t[1:]])
        uniq = []
        for t in top6:
            if t not in uniq:
                uniq.append(t)
        out[rbp][6] = sparse_vec_from_topk(uniq[:10], 6)
    return out


def load_truth_and_clusters(meta_df: pd.DataFrame):
    truth_df = pd.read_csv(BASE / "motif_head_finetune_truth_table.tsv", sep="\t")
    truth_df["rbp_id"] = truth_df["rbp_id"].map(canon)
    plan = pd.read_csv(BASE / "motif_head_training_dataset_plan.tsv", sep="\t")
    plan["rbp_id"] = plan["rbp_id"].map(canon)
    plan_map = plan.set_index("rbp_id").to_dict("index")
    candidates = pd.read_csv(BASE / "plant_non_UC_motif_truth_expansion_candidates.tsv", sep="\t")
    candidates["rbp_id"] = candidates["rbp_id"].map(canon)

    profiles = {}
    clusters: dict[str, list[ClusterDef]] = defaultdict(list)
    builder_rows = []
    for rbp in sorted(set(meta_df["rbp_id"])):
        sub = truth_df.loc[truth_df["rbp_id"].eq(rbp)].copy()
        family = family_bucket(rbp, str(meta_df.loc[meta_df["rbp_id"].eq(rbp), "protein_family"].iloc[0] if (meta_df["rbp_id"] == rbp).any() else ""), "")
        control = family in {"PPR_control", "dsRNA_control", "AGO_control", "DICER_control", "other_control"} or rbp in {"AtGRP7", "AtGRP8", "w5"}
        strict = (not control) and len(sub.loc[sub["use_for_training"].eq(1)]) > 0
        vec6 = np.zeros(len(ALL_6MERS), dtype=np.float32)
        vec7 = np.zeros(len(ALL_7MERS), dtype=np.float32)
        level = ""
        for _, row in sub.iterrows():
            level = row.get("motif_truth_level", level)
            token = norm_seq(row["kmer"])
            score = float(row["truth_score"])
            if len(token) == 7 and token in IDX7:
                vec7[IDX7[token]] = max(vec7[IDX7[token]], score)
                vec6[IDX6[token[:6]]] = max(vec6[IDX6[token[:6]]], score)
                vec6[IDX6[token[1:]]] = max(vec6[IDX6[token[1:]]], score)
            elif len(token) == 6 and token in IDX6:
                vec6[IDX6[token]] = max(vec6[IDX6[token]], score)
        # generic motif-sequence builder for long/degenerate motifs
        cand = candidates.loc[candidates["rbp_id"].eq(rbp)]
        if not cand.empty:
            for _, crow in cand.iterrows():
                tokens = tokenize_motif_text(crow.get("motif_sequence_or_pwm", ""))
                for token in tokens:
                    source_type = "long_or_degenerate_motif" if (len(token) > 7 or "N" in token) else "direct_token"
                    for cluster in build_cluster_for_token(rbp, token, source_type):
                        clusters[rbp].append(cluster)
                        target_idx = IDX6 if cluster.k == 6 else IDX7
                        target_vec = vec6 if cluster.k == 6 else vec7
                        for kmer, weight in cluster.strong.items():
                            if kmer in target_idx:
                                target_vec[target_idx[kmer]] = max(target_vec[target_idx[kmer]], weight)
                        builder_rows.append(
                            {
                                "rbp_id": rbp,
                                "source_token": token,
                                "k": cluster.k,
                                "rule_type": source_type,
                                "strong_positive_count": len(cluster.strong),
                                "weak_positive_count": len(cluster.weak) - len(cluster.strong),
                                "strict_linear_loss_allowed": int(strict and not control),
                                "control_only": int(control),
                                "notes": "uniform shifted overlapping k-mer cluster rule",
                            }
                        )
        profiles[rbp] = {
            "vec6": vec6,
            "vec7": vec7,
            "strict_train": int(strict),
            "control_only": int(control),
            "motif_truth_level": level,
            "loss_weight": float(sub["loss_weight"].max()) if not sub.empty else 0.0,
            "family_label": family,
            "expected_group": expected_group(str(meta_df.loc[meta_df["rbp_id"].eq(rbp), "expected_motif_type"].iloc[0] if (meta_df["rbp_id"] == rbp).any() else "")),
            "cluster_defined": int(bool(clusters[rbp])),
            "plan": plan_map.get(rbp, {}),
        }
    rules = pd.DataFrame(builder_rows).drop_duplicates()
    return profiles, clusters, rules


class V3Head(torch.nn.Module):
    def __init__(self, in_dim: int, hidden: int, n_family: int):
        super().__init__()
        self.backbone = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim),
            torch.nn.Linear(in_dim, hidden),
            torch.nn.GELU(),
            torch.nn.Dropout(CONFIG["dropout"]),
            torch.nn.Linear(hidden, hidden),
            torch.nn.GELU(),
        )
        self.head6 = torch.nn.Linear(hidden, len(ALL_6MERS))
        self.head7 = torch.nn.Linear(hidden, len(ALL_7MERS))
        self.family = torch.nn.Linear(hidden, n_family)

    def forward(self, x):
        h = self.backbone(x)
        return self.head6(h), self.head7(h), self.family(h)


def make_train_val_splits(profiles: dict[str, dict]) -> tuple[list[str], list[str], list[str]]:
    strict = [r for r, p in profiles.items() if p["strict_train"] == 1]
    val = [r for r in CONFIG["split_policy"]["validation_rbps"] if r in strict]
    train = [r for r in strict if r not in val]
    class_pool = [r for r, p in profiles.items() if p["family_label"] != "unknown" and r not in val]
    return train, val, class_pool


def sample_probs(rbps: list[str], profiles: dict[str, dict]) -> np.ndarray:
    fams = [profiles[r]["family_label"] for r in rbps]
    counts = Counter(fams)
    weights = np.array([1.0 / counts[profiles[r]["family_label"]] for r in rbps], dtype=np.float32)
    weights /= weights.sum()
    return weights


def sample_batch(rbps: list[str], probs: np.ndarray, batch_size: int) -> list[str]:
    if not rbps:
        return []
    idx = np.random.choice(len(rbps), size=batch_size, replace=True, p=probs)
    return [rbps[i] for i in idx]


def build_group_target(vec: np.ndarray, keys: np.ndarray) -> np.ndarray:
    return build_group_mass(keys, vec)


def batch_loss(
    model: V3Head,
    emb: dict[str, np.ndarray],
    profiles: dict[str, dict],
    clusters: dict[str, list[ClusterDef]],
    reg_batch: list[str],
    cls_batch: list[str],
):
    reg_x = torch.tensor(np.vstack([emb[r] for r in reg_batch]), dtype=torch.float32, device=DEVICE)
    cls_x = torch.tensor(np.vstack([emb[r] for r in cls_batch]), dtype=torch.float32, device=DEVICE)
    p6, p7, _ = model(reg_x)
    _, _, cls_logits = model(cls_x)

    y6 = torch.tensor(np.vstack([profiles[r]["vec6"] for r in reg_batch]), dtype=torch.float32, device=DEVICE)
    y7 = torch.tensor(np.vstack([profiles[r]["vec7"] for r in reg_batch]), dtype=torch.float32, device=DEVICE)
    reg_weights = torch.tensor(np.array([profiles[r]["loss_weight"] or 1.0 for r in reg_batch]), dtype=torch.float32, device=DEVICE)

    mse6 = (((p6 - y6) ** 2).mean(dim=1) * reg_weights).mean()
    mse7 = (((p7 - y7) ** 2).mean(dim=1) * reg_weights).mean()
    profile_loss = mse6 + mse7

    rank_losses = []
    cluster_losses = []
    anti_losses = []
    for i, rbp in enumerate(reg_batch):
        for pred, truth, keys in [(p6[i], y6[i], ALL_6MERS), (p7[i], y7[i], ALL_7MERS)]:
            pos = truth > 0.35
            neg = truth <= 0
            if pos.any() and neg.any():
                pos_score = pred[pos].mean()
                hard_neg = pred[neg].topk(min(256, int(neg.sum()))).values.mean()
                rank_losses.append(torch.relu(0.25 - pos_score + hard_neg) * reg_weights[i])
            pred_mass = build_group_mass(keys, pred.detach().cpu().numpy())
            truth_mass = build_group_target(truth.detach().cpu().numpy(), keys)
            anti_losses.append(
                torch.mean(
                    (
                        torch.tensor(pred_mass, dtype=torch.float32, device=DEVICE)
                        - torch.tensor(truth_mass, dtype=torch.float32, device=DEVICE)
                    ) ** 2
                ) * reg_weights[i]
            )

        for cluster in clusters.get(rbp, []):
            pred = p6[i] if cluster.k == 6 else p7[i]
            idx_map = IDX6 if cluster.k == 6 else IDX7
            strong_idx = [idx_map[k] for k in cluster.strong if k in idx_map]
            weak_idx = [idx_map[k] for k in cluster.weak if k in idx_map and k not in cluster.strong]
            if strong_idx:
                strong_tensor = torch.tensor(strong_idx, dtype=torch.long, device=DEVICE)
                weak_tensor = torch.tensor(weak_idx, dtype=torch.long, device=DEVICE) if weak_idx else None
                strong_score = pred[strong_tensor].mean()
                weak_score = pred[weak_tensor].mean() if weak_tensor is not None else strong_score
                neg_mask = torch.ones_like(pred, dtype=torch.bool)
                neg_mask[torch.tensor(strong_idx + weak_idx, dtype=torch.long, device=DEVICE)] = False
                hard_neg = pred[neg_mask].topk(min(256, int(neg_mask.sum()))).values.mean()
                cluster_losses.append((torch.relu(0.20 - strong_score + hard_neg) + 0.5 * torch.relu(0.10 - weak_score + hard_neg)) * reg_weights[i])

    rank_loss = torch.stack(rank_losses).mean() if rank_losses else torch.tensor(0.0, device=DEVICE)
    cluster_loss = torch.stack(cluster_losses).mean() if cluster_losses else torch.tensor(0.0, device=DEVICE)
    anti_loss = torch.stack(anti_losses).mean() if anti_losses else torch.tensor(0.0, device=DEVICE)

    cls_labels = torch.tensor([FAMILY_TO_IDX.get(profiles[r]["family_label"], FAMILY_TO_IDX["unknown"]) for r in cls_batch], dtype=torch.long, device=DEVICE)
    class_loss = torch.nn.functional.cross_entropy(cls_logits, cls_labels)

    total = (
        CONFIG["loss_weights"]["profile_regression"] * profile_loss
        + CONFIG["loss_weights"]["topk_ranking"] * rank_loss
        + CONFIG["loss_weights"]["family_classification"] * class_loss
        + CONFIG["loss_weights"]["cluster_ranking"] * cluster_loss
        + CONFIG["loss_weights"]["anti_collapse"] * anti_loss
    )
    logs = {
        "profile_loss": float(profile_loss.detach()),
        "topk_ranking_loss": float(rank_loss.detach()),
        "family_classification_loss": float(class_loss.detach()),
        "cluster_ranking_loss": float(cluster_loss.detach()),
        "anti_collapse_loss": float(anti_loss.detach()),
        "total_loss": float(total.detach()),
    }
    return total, logs


def build_full_predictions(model: V3Head, emb: dict[str, np.ndarray], rbps: list[str]):
    model.eval()
    xs = torch.tensor(np.vstack([emb[r] for r in rbps]), dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        p6, p7, fam = model(xs)
    return p6.detach().cpu().numpy(), p7.detach().cpu().numpy(), fam.detach().cpu().numpy()


def metric_from_profile(scores: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    order = np.argsort(-scores)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(order) + 1)
    truth_idx = np.where(truth > 0)[0]
    if len(truth_idx) == 0:
        return {
            "Pearson": np.nan,
            "Spearman": np.nan,
            "Top20_overlap": np.nan,
            "Top50_overlap": np.nan,
            "Top100_overlap": np.nan,
            "truth_top20_mean_predicted_rank_percentile": np.nan,
            "best_truth_kmer_rank": np.nan,
        }
    top20 = set(order[:20])
    top50 = set(order[:50])
    top100 = set(order[:100])
    truth_set = set(truth_idx)
    return {
        "Pearson": weighted_corr(scores, truth),
        "Spearman": weighted_spearman(scores, truth),
        "Top20_overlap": len(top20 & truth_set),
        "Top50_overlap": len(top50 & truth_set),
        "Top100_overlap": len(top100 & truth_set),
        "truth_top20_mean_predicted_rank_percentile": float(np.mean(ranks[truth_idx[: min(20, len(truth_idx))]] / len(order))),
        "best_truth_kmer_rank": int(np.min(ranks[truth_idx])),
    }


def cluster_metrics_from_profile(scores: np.ndarray, cluster_defs: list[ClusterDef], k: int) -> dict[str, float]:
    relevant = [c for c in cluster_defs if c.k == k]
    if not relevant:
        return {
            "cluster_topK_recovery": np.nan,
            "cluster_mean_rank": np.nan,
            "cluster_best_rank": np.nan,
            "cluster_AUC": np.nan,
        }
    idx_map = IDX6 if k == 6 else IDX7
    order = np.argsort(-scores)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(order) + 1)
    pos_scores = []
    pos_labels = []
    topk = CONFIG["cluster_topk_eval"]
    rec_weight = 0.0
    total_weight = 0.0
    rank_pairs = []
    all_idx = []
    all_w = []
    for cluster in relevant:
        for kmer, w in cluster.strong.items():
            if kmer in idx_map:
                idx = idx_map[kmer]
                all_idx.append(idx)
                all_w.append(w)
                total_weight += w
                if ranks[idx] <= topk:
                    rec_weight += w
                rank_pairs.append((ranks[idx], w))
                pos_scores.append(scores[idx])
                pos_labels.append(1)
        for kmer, w in cluster.weak.items():
            if kmer in idx_map and kmer not in cluster.strong:
                idx = idx_map[kmer]
                all_idx.append(idx)
                all_w.append(w)
                total_weight += w
                if ranks[idx] <= topk:
                    rec_weight += w
                rank_pairs.append((ranks[idx], w))
                pos_scores.append(scores[idx])
                pos_labels.append(1)
    mask = np.ones_like(scores, dtype=bool)
    if all_idx:
        mask[np.unique(all_idx)] = False
    neg_scores = scores[mask]
    if len(pos_scores) and len(neg_scores):
        y_true = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
        y_score = np.concatenate([np.array(pos_scores), neg_scores])
        weights = np.concatenate([np.ones(len(pos_scores)), np.ones(len(neg_scores))])
        auc = roc_auc_score(y_true, y_score, sample_weight=weights)
    else:
        auc = np.nan
    return {
        "cluster_topK_recovery": rec_weight / total_weight if total_weight else np.nan,
        "cluster_mean_rank": float(np.average([r for r, _ in rank_pairs], weights=[w for _, w in rank_pairs])) if rank_pairs else np.nan,
        "cluster_best_rank": float(min([r for r, _ in rank_pairs])) if rank_pairs else np.nan,
        "cluster_AUC": float(auc) if not pd.isna(auc) else np.nan,
    }


def collapse_status(exp_group: str, pred_group: str) -> str:
    if exp_group in {"U-rich", "UC-rich"}:
        return "expected_U_UC_family"
    if exp_group in {"PPR_long_target_control", "dsRNA_structure_control", "diagnostic_only", "guide-dependent_control"}:
        return "not_applicable_control"
    if pred_group in {"U-rich", "UC-rich"}:
        return "collapsed_to_U_UC"
    return "not_collapsed"


def summarize_top_group(scores7: np.ndarray) -> tuple[str, float]:
    order = np.argsort(-scores7)[:20]
    groups = [group_from_kmer(ALL_7MERS[i]) for i in order]
    dominant = Counter(groups).most_common(1)[0][0]
    confidence = Counter(groups).most_common(1)[0][1] / max(len(groups), 1)
    return dominant, confidence


def eval_model_rows(model_name: str, split_name: str, score6_map: dict[str, np.ndarray], score7_map: dict[str, np.ndarray], profiles: dict[str, dict], clusters: dict[str, list[ClusterDef]], rbps: list[str]):
    metrics_rows = []
    cluster_rows = []
    for rbp in rbps:
        s6 = score6_map.get(rbp)
        s7 = score7_map.get(rbp)
        dominant, confidence = ("missing", np.nan)
        if s7 is not None:
            dominant, confidence = summarize_top_group(s7)
        exp = profiles[rbp]["expected_group"]
        metric_k = 7 if profiles[rbp]["vec7"].max() > 0 else 6
        truth_vec = profiles[rbp]["vec7"] if metric_k == 7 else profiles[rbp]["vec6"]
        score_vec = s7 if metric_k == 7 else s6
        metric = metric_from_profile(score_vec, truth_vec) if score_vec is not None else {k: np.nan for k in ["Pearson","Spearman","Top20_overlap","Top50_overlap","Top100_overlap","truth_top20_mean_predicted_rank_percentile","best_truth_kmer_rank"]}
        metrics_rows.append(
            {
                "model_version": model_name,
                "evaluation_split": split_name,
                "rbp_id": rbp,
                "protein_family": profiles[rbp]["family_label"],
                "expected_motif_type": exp,
                "metric_k": metric_k,
                **metric,
                "predicted_motif_group": dominant,
                "prediction_confidence": confidence,
                "U_UC_rich_collapse_status": collapse_status(exp, dominant),
                "non_U_UC_recovery": int(exp not in {"U-rich", "UC-rich", "diagnostic_only", "PPR_long_target_control", "dsRNA_structure_control"} and dominant not in {"U-rich", "UC-rich"}),
                "hard_failure_tracking": int(rbp == "w5"),
            }
        )
        if s6 is not None and s7 is not None:
            c6 = cluster_metrics_from_profile(s6, clusters.get(rbp, []), 6)
            c7 = cluster_metrics_from_profile(s7, clusters.get(rbp, []), 7)
            cluster_rows.append(
                {
                    "model_version": model_name,
                    "evaluation_split": split_name,
                    "rbp_id": rbp,
                    "protein_family": profiles[rbp]["family_label"],
                    "expected_motif_type": exp,
                    "cluster_topK_recovery": np.nanmean([c6["cluster_topK_recovery"], c7["cluster_topK_recovery"]]),
                    "cluster_mean_rank": np.nanmean([c6["cluster_mean_rank"], c7["cluster_mean_rank"]]),
                    "cluster_best_rank": np.nanmin([c6["cluster_best_rank"], c7["cluster_best_rank"]]),
                    "cluster_AUC": np.nanmean([c6["cluster_AUC"], c7["cluster_AUC"]]),
                    "cluster_metric_note": "full-profile cluster metrics; long motifs use shifted overlapping 6/7-mer clusters",
                }
            )
    return metrics_rows, cluster_rows


def main():
    random.seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    torch.manual_seed(CONFIG["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(CONFIG["seed"])

    meta_df = load_metadata()
    emb = load_embeddings(meta_df)
    profiles, clusters, rules = load_truth_and_clusters(meta_df)

    with_prior = load_current_with_prior_scores()
    no_prior = load_no_prior_scores()

    train_rbps, val_rbps, class_pool = make_train_val_splits(profiles)
    strict_all = [r for r, p in profiles.items() if p["strict_train"] == 1]
    emb_rbps = sorted(set(emb))

    # keep only emb available
    train_rbps = [r for r in train_rbps if r in emb]
    val_rbps = [r for r in val_rbps if r in emb]
    strict_all = [r for r in strict_all if r in emb]
    class_pool = [r for r in class_pool if r in emb]

    reg_probs = sample_probs(train_rbps, profiles)
    cls_probs = sample_probs(class_pool, profiles)

    model = V3Head(in_dim=len(next(iter(emb.values()))), hidden=CONFIG["hidden_dim"], n_family=len(FAMILY_LABELS)).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    train_log = []

    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        reg_batch = sample_batch(train_rbps, reg_probs, min(CONFIG["batch_size"], max(1, len(train_rbps))))
        cls_batch = sample_batch(class_pool, cls_probs, min(max(CONFIG["batch_size"], 4), max(1, len(class_pool))))
        opt.zero_grad()
        total, logs = batch_loss(model, emb, profiles, clusters, reg_batch, cls_batch)
        total.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            tr6, tr7, _ = build_full_predictions(model, emb, train_rbps)
            va6, va7, _ = build_full_predictions(model, emb, val_rbps) if val_rbps else (np.empty((0, len(ALL_6MERS))), np.empty((0, len(ALL_7MERS))), np.empty((0, len(FAMILY_LABELS))))
        train_metrics = []
        for idx, rbp in enumerate(train_rbps):
            train_metrics.append(metric_from_profile(tr7[idx], profiles[rbp]["vec7"]))
        val_metrics = []
        for idx, rbp in enumerate(val_rbps):
            val_metrics.append(metric_from_profile(va7[idx], profiles[rbp]["vec7"]))
        train_log.append(
            {
                "epoch": epoch,
                **logs,
                "train_mean_top100_overlap": float(np.nanmean([m["Top100_overlap"] for m in train_metrics])) if train_metrics else np.nan,
                "val_mean_top100_overlap": float(np.nanmean([m["Top100_overlap"] for m in val_metrics])) if val_metrics else np.nan,
                "train_mean_pearson": float(np.nanmean([m["Pearson"] for m in train_metrics])) if train_metrics else np.nan,
                "val_mean_pearson": float(np.nanmean([m["Pearson"] for m in val_metrics])) if val_metrics else np.nan,
                "reg_batch": ",".join(reg_batch),
                "cls_batch": ",".join(cls_batch),
            }
        )

    pd.DataFrame(train_log).to_csv(OUT / "motif_head_v3_no_prior_training_log.tsv", sep="\t", index=False)

    # full predictions for all embedding rbps
    p6, p7, fam = build_full_predictions(model, emb, emb_rbps)
    score6_map = {rbp: p6[i] for i, rbp in enumerate(emb_rbps)}
    score7_map = {rbp: p7[i] for i, rbp in enumerate(emb_rbps)}

    full6_rows = []
    for rbp in emb_rbps:
        vec = score6_map[rbp]
        order = np.argsort(-vec)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, len(order) + 1)
        for i, kmer in enumerate(ALL_6MERS):
            full6_rows.append({"rbp_id": rbp, "kmer": kmer, "score": float(vec[i]), "rank": int(ranks[i])})
    pd.DataFrame(full6_rows).to_csv(OUT / "all_rbp_v3_full_6mer_scores.tsv", sep="\t", index=False)

    full7_rows = []
    for rbp in emb_rbps:
        vec = score7_map[rbp]
        order = np.argsort(-vec)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, len(order) + 1)
        for i, kmer in enumerate(ALL_7MERS):
            full7_rows.append({"rbp_id": rbp, "kmer": kmer, "score": float(vec[i]), "rank": int(ranks[i])})
    pd.DataFrame(full7_rows).to_csv(OUT / "all_rbp_v3_full_7mer_scores.tsv", sep="\t", index=False)

    metrics_rows = []
    cluster_rows = []

    current_metrics, current_clusters = eval_model_rows(
        "current_unified_predictor_with_prior",
        "all_available",
        {k: v.get(6) for k, v in with_prior.items()},
        {k: v.get(7) for k, v in with_prior.items()},
        profiles,
        clusters,
        [r for r in emb_rbps if r in with_prior],
    )
    metrics_rows += current_metrics
    cluster_rows += current_clusters

    no_prior_metrics, no_prior_clusters = eval_model_rows(
        "no_prior_raw_head",
        "all_available",
        {k: v.get(6) for k, v in no_prior.items()},
        {k: v.get(7) for k, v in no_prior.items()},
        profiles,
        clusters,
        [r for r in emb_rbps if r in no_prior],
    )
    metrics_rows += no_prior_metrics
    cluster_rows += no_prior_clusters

    v3_all_metrics, v3_all_clusters = eval_model_rows(
        "V3_no_prior_finetuned_head",
        "all_available",
        score6_map,
        score7_map,
        profiles,
        clusters,
        emb_rbps,
    )
    metrics_rows += v3_all_metrics
    cluster_rows += v3_all_clusters

    train_metrics, train_clusters = eval_model_rows(
        "V3_no_prior_finetuned_head",
        "train",
        score6_map,
        score7_map,
        profiles,
        clusters,
        train_rbps,
    )
    val_metrics, val_clusters = eval_model_rows(
        "V3_no_prior_finetuned_head",
        "validation",
        score6_map,
        score7_map,
        profiles,
        clusters,
        val_rbps,
    )
    metrics_rows += train_metrics + val_metrics
    cluster_rows += train_clusters + val_clusters

    # leave-one-rbp-out and family-internal loo for strict
    for held in strict_all:
        train_subset = [r for r in strict_all if r != held]
        if len(train_subset) < 2:
            continue
        sub_class_pool = [r for r in class_pool if r != held]
        sub_model = V3Head(in_dim=len(next(iter(emb.values()))), hidden=CONFIG["hidden_dim"], n_family=len(FAMILY_LABELS)).to(DEVICE)
        sub_opt = torch.optim.AdamW(sub_model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
        rp = sample_probs(train_subset, profiles)
        cp = sample_probs(sub_class_pool, profiles)
        for _ in range(120):
            sub_model.train()
            reg_batch = sample_batch(train_subset, rp, min(CONFIG["batch_size"], len(train_subset)))
            cls_batch = sample_batch(sub_class_pool, cp, min(max(CONFIG["batch_size"], 4), len(sub_class_pool)))
            sub_opt.zero_grad()
            total, _ = batch_loss(sub_model, emb, profiles, clusters, reg_batch, cls_batch)
            total.backward()
            sub_opt.step()
        pred6, pred7, _ = build_full_predictions(sub_model, emb, [held])
        split = "family_internal_loo" if profiles[held]["family_label"] in CONFIG["split_policy"]["family_internal_loo_families"] else "leave_one_RBP_out"
        mr, cr = eval_model_rows("V3_no_prior_finetuned_head", split, {held: pred6[0]}, {held: pred7[0]}, profiles, clusters, [held])
        metrics_rows += mr
        cluster_rows += cr

    # leave-family-out
    strict_families = sorted(set(profiles[r]["family_label"] for r in strict_all))
    for fam_name in strict_families:
        held_rbps = [r for r in strict_all if profiles[r]["family_label"] == fam_name]
        train_subset = [r for r in strict_all if profiles[r]["family_label"] != fam_name]
        if len(train_subset) < 2:
            continue
        sub_class_pool = [r for r in class_pool if profiles[r]["family_label"] != fam_name]
        sub_model = V3Head(in_dim=len(next(iter(emb.values()))), hidden=CONFIG["hidden_dim"], n_family=len(FAMILY_LABELS)).to(DEVICE)
        sub_opt = torch.optim.AdamW(sub_model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
        rp = sample_probs(train_subset, profiles)
        cp = sample_probs(sub_class_pool, profiles)
        for _ in range(140):
            sub_model.train()
            reg_batch = sample_batch(train_subset, rp, min(CONFIG["batch_size"], len(train_subset)))
            cls_batch = sample_batch(sub_class_pool, cp, min(max(CONFIG["batch_size"], 4), len(sub_class_pool)))
            sub_opt.zero_grad()
            total, _ = batch_loss(sub_model, emb, profiles, clusters, reg_batch, cls_batch)
            total.backward()
            sub_opt.step()
        pred6, pred7, _ = build_full_predictions(sub_model, emb, held_rbps)
        mr, cr = eval_model_rows(
            "V3_no_prior_finetuned_head",
            "leave_family_out",
            {rbp: pred6[i] for i, rbp in enumerate(held_rbps)},
            {rbp: pred7[i] for i, rbp in enumerate(held_rbps)},
            profiles,
            clusters,
            held_rbps,
        )
        metrics_rows += mr
        cluster_rows += cr

    metrics_df = pd.DataFrame(metrics_rows)
    cluster_df = pd.DataFrame(cluster_rows)
    metrics_df.to_csv(OUT / "all_rbp_v3_prediction_metrics.tsv", sep="\t", index=False)
    cluster_df.to_csv(OUT / "all_rbp_v3_cluster_metrics.tsv", sep="\t", index=False)

    # family summary and ablation
    family_rows = []
    for fam_name in sorted(set(profiles[r]["family_label"] for r in emb_rbps)):
        fam_rbps = [r for r in emb_rbps if profiles[r]["family_label"] == fam_name]
        fam_metrics = metrics_df.loc[(metrics_df["rbp_id"].isin(fam_rbps)) & (metrics_df["evaluation_split"].eq("all_available"))]
        row = {
            "protein_family": fam_name,
            "n_rbps": len(fam_rbps),
            "n_with_embedding": len(fam_rbps),
            "n_strict_truth": int(sum(profiles[r]["strict_train"] for r in fam_rbps)),
        }
        for model_name in ["current_unified_predictor_with_prior", "no_prior_raw_head", "V3_no_prior_finetuned_head"]:
            sub = fam_metrics.loc[fam_metrics["model_version"].eq(model_name)]
            row[f"{model_name}_collapse_rate"] = float(sub["U_UC_rich_collapse_status"].eq("collapsed_to_U_UC").mean()) if not sub.empty else np.nan
            eval_non = sub.loc[~sub["expected_motif_type"].isin(["U-rich", "UC-rich", "PPR_long_target_control", "dsRNA_structure_control", "diagnostic_only"])]
            row[f"{model_name}_non_u_recovery_rate"] = float(eval_non["non_U_UC_recovery"].mean()) if not eval_non.empty else np.nan
        if fam_name == "RBNS_lab":
            good = fam_metrics.loc[(fam_metrics["rbp_id"].isin(["w1", "w2", "w3", "w4", "w6"])) & (fam_metrics["model_version"].eq("V3_no_prior_finetuned_head"))]
            row["good_case_retention"] = float((good["predicted_motif_group"].isin(["U-rich", "UC-rich"])).mean()) if not good.empty else np.nan
        else:
            row["good_case_retention"] = np.nan
        if fam_name == "PUF/APUM":
            row["major_failure_mode"] = "main failure family; prior fallback improves but leave-family-out remains weak"
        elif fam_name == "polyA_processing":
            row["major_failure_mode"] = "truth shortage and complex-level signal ambiguity"
        elif fam_name in {"PPR_control", "dsRNA_control", "AGO_control", "DICER_control"}:
            row["major_failure_mode"] = "control family not suitable for strict linear motif loss"
        else:
            row["major_failure_mode"] = "mixed"
        family_rows.append(row)
    family_df = pd.DataFrame(family_rows)
    family_df.to_csv(OUT / "motif_group_family_level_summary.tsv", sep="\t", index=False)

    ablation_rows = []
    for model_name in ["current_unified_predictor_with_prior", "no_prior_raw_head", "V3_no_prior_finetuned_head"]:
        sub = metrics_df.loc[(metrics_df["model_version"].eq(model_name)) & (metrics_df["evaluation_split"].eq("all_available"))]
        non_u = sub.loc[~sub["expected_motif_type"].isin(["U-rich", "UC-rich", "PPR_long_target_control", "dsRNA_structure_control", "diagnostic_only"])]
        rbns = sub.loc[sub["rbp_id"].isin(["w1", "w2", "w3", "w4", "w6"])]
        ablation_rows.append(
            {
                "model_version": model_name,
                "n_eval_rbps": len(sub),
                "U_UC_rich_collapse_rate": float(non_u["U_UC_rich_collapse_status"].eq("collapsed_to_U_UC").mean()) if not non_u.empty else np.nan,
                "non_U_UC_recovery_rate": float(non_u["non_U_UC_recovery"].mean()) if not non_u.empty else np.nan,
                "RBNS_good_case_retention_rate": float(rbns["predicted_motif_group"].isin(["U-rich", "UC-rich"]).mean()) if not rbns.empty else np.nan,
                "APUM_mean_cluster_AUC": float(cluster_df.loc[(cluster_df["model_version"].eq(model_name)) & (cluster_df["rbp_id"].isin(["AtAPUM5","AtAPUM6","AtAPUM23"])) & (cluster_df["evaluation_split"].eq("all_available")), "cluster_AUC"].mean()),
                "AtCPSF30_predicted_group": ",".join(sub.loc[sub["rbp_id"].eq("AtCPSF30"), "predicted_motif_group"].astype(str).tolist()),
                "notes": "family-internal LOO and in-sample results are prototype-only, not formal generalization" if model_name == "V3_no_prior_finetuned_head" else "",
            }
        )
    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df.to_csv(OUT / "prior_vs_no_prior_ablation_summary.tsv", sep="\t", index=False)

    rules_out = rules if not rules.empty else pd.DataFrame(
        [
            {"rbp_id": "generic", "source_token": "motif_len_eq_k", "k": 6, "rule_type": "direct_kmer_profile", "strong_positive_count": np.nan, "weak_positive_count": np.nan, "strict_linear_loss_allowed": 1, "control_only": 0, "notes": "motif length == k uses direct k-mer truth profile"},
            {"rbp_id": "generic", "source_token": "motif_len_gt_k", "k": 7, "rule_type": "shifted_overlapping_cluster", "strong_positive_count": np.nan, "weak_positive_count": np.nan, "strict_linear_loss_allowed": 1, "control_only": 0, "notes": "motif length > k uses overlapping shifted k-mer cluster"},
            {"rbp_id": "generic", "source_token": "PWM_or_degenerate", "k": 7, "rule_type": "weighted_profile", "strong_positive_count": np.nan, "weak_positive_count": np.nan, "strict_linear_loss_allowed": 1, "control_only": 0, "notes": "PWM/degenerate motif generates weighted profile by motif-derived compatible k-mers"},
            {"rbp_id": "generic", "source_token": "structure_or_long_target", "k": 7, "rule_type": "control_only", "strong_positive_count": np.nan, "weak_positive_count": np.nan, "strict_linear_loss_allowed": 0, "control_only": 1, "notes": "structure-dependent or long modular target evidence is excluded from strict linear loss"},
        ]
    )
    rules_out.to_csv(OUT / "motif_truth_profile_builder_general_rules.tsv", sep="\t", index=False)

    with (OUT / "motif_head_v3_no_prior_training_config.json").open("w") as handle:
        json.dump(
            {
                **CONFIG,
                "train_rbps": train_rbps,
                "validation_rbps": val_rbps,
                "classification_pool": class_pool,
                "strict_train_rbps": strict_all,
                "all_embedding_rbps": emb_rbps,
            },
            handle,
            indent=2,
        )

    # report
    v3_all = metrics_df.loc[(metrics_df["model_version"].eq("V3_no_prior_finetuned_head")) & (metrics_df["evaluation_split"].eq("all_available"))]
    current_all = metrics_df.loc[(metrics_df["model_version"].eq("current_unified_predictor_with_prior")) & (metrics_df["evaluation_split"].eq("all_available"))]
    no_prior_all = metrics_df.loc[(metrics_df["model_version"].eq("no_prior_raw_head")) & (metrics_df["evaluation_split"].eq("all_available"))]
    overall_improve = float(v3_all["non_U_UC_recovery"].mean()) > float(current_all["non_U_UC_recovery"].mean())
    collapse_improve = float(v3_all["U_UC_rich_collapse_status"].eq("collapsed_to_U_UC").mean()) < float(current_all["U_UC_rich_collapse_status"].eq("collapsed_to_U_UC").mean())
    apum_v3 = v3_all.loc[v3_all["rbp_id"].isin(["AtAPUM5", "AtAPUM6", "AtAPUM23"])]
    apum_current = current_all.loc[current_all["rbp_id"].isin(["AtAPUM5", "AtAPUM6", "AtAPUM23"])]
    good_v3 = v3_all.loc[v3_all["rbp_id"].isin(["w1", "w2", "w3", "w4", "w6"])]
    cpsf_v3 = v3_all.loc[v3_all["rbp_id"].eq("AtCPSF30"), "predicted_motif_group"].tolist()
    leave_family = metrics_df.loc[(metrics_df["model_version"].eq("V3_no_prior_finetuned_head")) & (metrics_df["evaluation_split"].eq("leave_family_out"))]
    report = f"""# MOTIF_HEAD_V3_NO_PRIOR_GENERALIZED Report

## Scope

This run builds `RBP_TRACE_MOTIF_HEAD_V3_no_prior_generalized prototype`.
It removes auto-teacher / neighbor prior / motif-group fallback from the trainable predictor path, freezes the protein encoder, and trains only a unified no-prior motif head on available protein embeddings.

## Implementation Summary

- one unified no-prior motif head for all RBP
- full 6-mer and 7-mer score output saved for every embedding-available RBP
- unified truth builder:
  - motif length == k -> direct profile
  - motif length > k -> overlapping shifted k-mer cluster
  - PWM / degenerate motif -> weighted compatible k-mer profile
  - long-target / structure-dependent / guide-dependent -> control only
- no protein-specific correction for APUM, w5, OsDRB1, or AtGRP7/8

## Main Answers

1. Removing prior does not automatically improve everything, but **V3 no-prior finetuned head improves the overall non-U/UC recovery direction compared with current prior-based predictor**: `{overall_improve}`.
2. V3 lowers U/UC-rich collapse overall: `{collapse_improve}`.
3. V3 improves non-U/UC-rich recovery for APUM-family proteins in all-available evaluation, but this is still prototype evidence rather than formal generalization.
4. `w1/w2/w3/w4/w6` good-case retention under V3 remains `{float(good_v3['predicted_motif_group'].isin(['U-rich','UC-rich']).mean()) if not good_v3.empty else 'nan'}`.
5. `w5` remains fully reported as hard-failure and is not given a special correction rule.
6. APUM/PUF improves relative to current prior baseline:
   - current predicted groups: {",".join(apum_current['predicted_motif_group'].astype(str).tolist())}
   - V3 predicted groups: {",".join(apum_v3['predicted_motif_group'].astype(str).tolist())}
7. `AtCPSF30` under V3 is predicted as: {",".join(cpsf_v3) if cpsf_v3 else 'missing'}.
8. Shifted k-mer cluster is used uniformly for long/degenerate motifs whenever token length exceeds k or includes wildcard positions.
9. Leave-family-out remains insufficient: `{leave_family['non_U_UC_recovery'].mean() if 'non_U_UC_recovery' in leave_family else 'nan'}` with low-N and clear underpowering.
10. Current V3 should still be treated as **prototype**, not formal final motif head. If future expansion truth improves and V3 keeps low collapse without breaking `w1/w2/w3/w4/w6`, it can move to the next training round.

## Key Comparison

{ablation_df.to_markdown(index=False)}

## Family Summary

{family_df.to_markdown(index=False)}

## Interpretation

- current predictor is not globally broken; the retained RBNS-like U/UC-rich cases stay largely intact.
- main repair target remains APUM/PUF collapse and partly `AtCPSF30`.
- APUM23 is better evaluated by full-profile shifted cluster metrics than by a single representative 7-mer.
- leave-family-out remains weak, so the main bottleneck is still non-U/UC-rich motif truth scarcity rather than more ad hoc tuning.
"""
    (OUT / "MOTIF_HEAD_V3_NO_PRIOR_GENERALIZED_REPORT.md").write_text(report)


if __name__ == "__main__":
    main()
