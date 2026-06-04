#!/usr/bin/env python3
"""Predict motif preference and motif-aware RNA-window binding for external RBPs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from unified_config import get_config
from unified_auto_teacher import build_auto_teacher_outputs
from unified_original_logic_model import OriginalLogicUnifiedRBPModel


RNA_BYTE_MAP = np.full(256, 4, dtype=np.uint8)
RNA_BYTE_MAP[ord("A")] = 0
RNA_BYTE_MAP[ord("a")] = 0
RNA_BYTE_MAP[ord("C")] = 1
RNA_BYTE_MAP[ord("c")] = 1
RNA_BYTE_MAP[ord("G")] = 2
RNA_BYTE_MAP[ord("g")] = 2
RNA_BYTE_MAP[ord("U")] = 3
RNA_BYTE_MAP[ord("u")] = 3
RNA_BYTE_MAP[ord("T")] = 3
RNA_BYTE_MAP[ord("t")] = 3

PLANT_SPECIES = {"ARATH", "ORYSJ", "ORYSA", "MAIZE", "SOLLC"}
REQUIRED_WINDOW_COLUMNS = ["gene_id", "transcript_id", "window_start", "window_end", "rna_seq"]


def clean_protein(seq: str) -> str:
    return str(seq).upper().replace("*", "")


def phys_feats(seq: str) -> List[float]:
    seq = clean_protein(seq)
    n = max(len(seq), 1)
    return [len(seq) / 1000.0, seq.count("G") / n, seq.count("R") / n, seq.count("K") / n]


def kingdom_label_from_query_id(query_protein_id: str) -> int:
    prefix = str(query_protein_id).split("|", 1)[0].upper()
    return 1 if prefix in PLANT_SPECIES else 0


def read_fasta(path: Path) -> Dict[str, str]:
    records: Dict[str, str] = {}
    header = None
    parts: List[str] = []
    opener = open
    if path.suffix == ".gz":
        import gzip

        opener = gzip.open
    with opener(path, "rt", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records[header] = "".join(parts)
                header = line[1:].split()[0]
                parts = []
            else:
                parts.append(line)
    if header is not None:
        records[header] = "".join(parts)
    return records


def encode_rna_sequences(seqs: Iterable[str], rna_len: int) -> np.ndarray:
    seqs = list(seqs)
    arr = np.full((len(seqs), rna_len), 4, dtype=np.uint8)
    for i, seq in enumerate(seqs):
        s = str(seq).strip()
        if not s:
            continue
        b = np.frombuffer(s.encode("ascii", errors="ignore"), dtype=np.uint8)
        if len(b) == 0:
            continue
        codes = RNA_BYTE_MAP[b[:rna_len]]
        arr[i, : len(codes)] = codes
    return arr


def topk_indices_desc(values: np.ndarray, k: int) -> np.ndarray:
    if k >= len(values):
        return np.argsort(-values)
    idx = np.argpartition(-values, k - 1)[:k]
    return idx[np.argsort(-values[idx])]


def topk_indices_asc(values: np.ndarray, k: int) -> np.ndarray:
    if k >= len(values):
        return np.argsort(values)
    idx = np.argpartition(values, k - 1)[:k]
    return idx[np.argsort(values[idx])]


def parse_query_ids(raw: str | None, cfg) -> List[str]:
    available = list(cfg.rbp_id_to_query_protein_id.keys())
    if not raw:
        return available
    requested = []
    for token in raw.split(","):
        value = token.strip()
        if value:
            requested.append(value)
    requested = list(dict.fromkeys(requested))
    missing = [x for x in requested if x not in cfg.rbp_id_to_query_protein_id]
    if missing:
        raise ValueError(f"unknown rbp_id(s): {missing}. Available: {available}")
    return requested


def load_window_table(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", compression="infer")
    missing = [c for c in REQUIRED_WINDOW_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"window table missing columns: {missing}")
    frame = frame.copy()
    frame["window_start"] = frame["window_start"].astype(int)
    frame["window_end"] = frame["window_end"].astype(int)
    frame["rna_seq"] = frame["rna_seq"].astype(str).str.upper().str.replace("T", "U", regex=False)
    if "rbp_id" in frame.columns:
        frame["rbp_id"] = frame["rbp_id"].astype(str)
    return frame


def normalize_rna_for_fingerprint(seq: str) -> str:
    seq = str(seq).replace(" ", "").replace("\n", "").upper().replace("T", "U")
    allowed = set("ACGUN")
    return "".join(base if base in allowed else "N" for base in seq)


def compute_rna_row_order_sha256(seqs: Sequence[str]) -> str:
    hasher = hashlib.sha256()
    for seq in seqs:
        hasher.update(normalize_rna_for_fingerprint(seq).encode("ascii"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def load_structure_cache_for_frame(
    frame: pd.DataFrame,
    structure_npy: Path,
    structure_meta_json: Path,
) -> Tuple[np.ndarray, Dict[str, object]]:
    with open(structure_meta_json, "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    structure = np.load(structure_npy, mmap_mode="r")
    if structure.ndim != 2:
        raise ValueError(f"structure cache must be 2D: {structure_npy}")
    if int(meta.get("n_rows", structure.shape[0])) != int(structure.shape[0]):
        raise ValueError("structure meta n_rows does not match structure cache array shape")
    if int(structure.shape[0]) != int(len(frame)):
        raise ValueError(
            "structure cache row count does not match window table row count: "
            f"{structure.shape[0]} vs {len(frame)}"
        )
    expected_rna_len = int(meta.get("rna_len", structure.shape[1]))
    if int(structure.shape[1]) != expected_rna_len:
        raise ValueError("structure cache width does not match structure meta rna_len")
    observed_lengths = frame["rna_seq"].astype(str).str.len().unique().tolist()
    if len(observed_lengths) != 1 or int(observed_lengths[0]) != expected_rna_len:
        raise ValueError(
            "window RNA length does not match structure cache rna_len: "
            f"observed={observed_lengths} expected={expected_rna_len}"
        )
    expected_hash = meta.get("rna_row_order_sha256")
    if expected_hash:
        observed_hash = compute_rna_row_order_sha256(frame["rna_seq"].tolist())
        if observed_hash != expected_hash:
            raise ValueError(
                "window row order does not match structure cache row-order fingerprint. "
                "Rebuild the structure cache from the exact same window order."
            )
    return structure, meta


def load_override_motif_profiles(
    profile_csv: Path,
    kmers: np.ndarray,
    query_ids: Sequence[str],
    cfg,
    top_k: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, np.ndarray]]:
    frame = pd.read_csv(profile_csv, sep=None, engine="python")
    if "kmer" in frame.columns:
        kmer_col = "kmer"
    else:
        kmer_col = frame.columns[0]
        frame = frame.rename(columns={kmer_col: "kmer"})
        kmer_col = "kmer"
    frame["kmer"] = frame[kmer_col].astype(str).str.upper().str.replace("T", "U", regex=False)
    frame = frame.drop_duplicates(subset=["kmer"], keep="first").set_index("kmer")
    aligned = frame.reindex([str(k).upper().replace("T", "U") for k in kmers.tolist()])
    if aligned.isna().all(axis=None):
        raise ValueError(f"override motif profile file has no overlapping k-mers with motif_profile_npz: {profile_csv}")

    top_rows = []
    summary_rows = []
    motif_profiles: Dict[str, np.ndarray] = {}
    for rbp_id in query_ids:
        query_protein_id = cfg.rbp_id_to_query_protein_id[rbp_id]
        candidate_cols = [rbp_id, query_protein_id]
        values = None
        chosen_col = None
        for col in candidate_cols:
            if col in aligned.columns:
                values = aligned[col].to_numpy(dtype=np.float32)
                chosen_col = col
                break
        if values is None:
            raise ValueError(
                f"override motif profile missing column for {rbp_id}. "
                f"Tried columns: {candidate_cols}. Available sample columns: {aligned.columns[:10].tolist()}"
            )
        values = np.nan_to_num(values, nan=0.0, posinf=10.0, neginf=-10.0).astype(np.float32)
        motif_profiles[rbp_id] = values
        pos_idx = topk_indices_desc(values, top_k)
        neg_idx = topk_indices_asc(values, top_k)
        pos_kmers = []
        neg_kmers = []
        for rank, idx in enumerate(pos_idx, start=1):
            kmer = str(kmers[idx])
            score = float(values[idx])
            pos_kmers.append(kmer)
            top_rows.append(
                {
                    "rbp_id": rbp_id,
                    "query_protein_id": query_protein_id,
                    "source_profile_column": chosen_col,
                    "direction": "positive",
                    "rank": rank,
                    "kmer": kmer,
                    "predicted_zscore": score,
                }
            )
        for rank, idx in enumerate(neg_idx, start=1):
            kmer = str(kmers[idx])
            score = float(values[idx])
            neg_kmers.append(kmer)
            top_rows.append(
                {
                    "rbp_id": rbp_id,
                    "query_protein_id": query_protein_id,
                    "source_profile_column": chosen_col,
                    "direction": "negative",
                    "rank": rank,
                    "kmer": kmer,
                    "predicted_zscore": score,
                }
            )
        summary_rows.append(
            {
                "rbp_id": rbp_id,
                "query_protein_id": query_protein_id,
                "source_profile_column": chosen_col,
                "protein_length": float("nan"),
                "top_positive_kmers": ",".join(pos_kmers[:10]),
                "top_negative_kmers": ",".join(neg_kmers[:10]),
                "max_predicted_zscore": float(values[pos_idx[0]]),
                "min_predicted_zscore": float(values[neg_idx[0]]),
            }
        )
    return pd.DataFrame(top_rows), pd.DataFrame(summary_rows), motif_profiles


def summarize_profile_dict(
    profile_map: Dict[str, np.ndarray],
    cfg,
    kmers: np.ndarray,
    top_k: int,
    extra_summary: pd.DataFrame | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    top_rows = []
    summary_rows = []
    extra_lookup = {}
    if extra_summary is not None and not extra_summary.empty and "rbp_id" in extra_summary.columns:
        extra_lookup = extra_summary.set_index("rbp_id").to_dict(orient="index")
    for rbp_id, values in profile_map.items():
        query_protein_id = cfg.rbp_id_to_query_protein_id[rbp_id]
        pos_idx = topk_indices_desc(values, top_k)
        neg_idx = topk_indices_asc(values, top_k)
        pos_kmers = []
        neg_kmers = []
        for rank, idx in enumerate(pos_idx, start=1):
            kmer = str(kmers[idx])
            score = float(values[idx])
            pos_kmers.append(kmer)
            top_rows.append(
                {
                    "rbp_id": rbp_id,
                    "query_protein_id": query_protein_id,
                    "direction": "positive",
                    "rank": rank,
                    "kmer": kmer,
                    "predicted_zscore": score,
                }
            )
        for rank, idx in enumerate(neg_idx, start=1):
            kmer = str(kmers[idx])
            score = float(values[idx])
            neg_kmers.append(kmer)
            top_rows.append(
                {
                    "rbp_id": rbp_id,
                    "query_protein_id": query_protein_id,
                    "direction": "negative",
                    "rank": rank,
                    "kmer": kmer,
                    "predicted_zscore": score,
                }
            )
        row = {
            "rbp_id": rbp_id,
            "query_protein_id": query_protein_id,
            "protein_length": float("nan"),
            "top_positive_kmers": ",".join(pos_kmers[:10]),
            "top_negative_kmers": ",".join(neg_kmers[:10]),
            "max_predicted_zscore": float(values[pos_idx[0]]),
            "min_predicted_zscore": float(values[neg_idx[0]]),
        }
        row.update(extra_lookup.get(rbp_id, {}))
        summary_rows.append(row)
    return pd.DataFrame(top_rows), pd.DataFrame(summary_rows)


def load_query_embedding_table(query_embedding_npy: Path, query_index_tsv: Path, train_embedding_npy: Path):
    train_embeddings = np.load(train_embedding_npy).astype(np.float32)
    mean = train_embeddings.mean(axis=0, keepdims=True)
    std = train_embeddings.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    query_embeddings = np.load(query_embedding_npy).astype(np.float32)
    query_index = pd.read_csv(query_index_tsv, sep="\t")
    if "embedding_row" not in query_index.columns:
        query_index["embedding_row"] = np.arange(len(query_index), dtype=np.int64)
    if len(query_index) != query_embeddings.shape[0]:
        raise ValueError("query embedding index row count does not match query embedding matrix")
    normalized_queries = ((query_embeddings - mean) / std).astype(np.float32)
    return normalized_queries, query_index


def resolve_query_embedding_row(query_index: pd.DataFrame, query_protein_id: str) -> int:
    key_col = "rbp_id" if "rbp_id" in query_index.columns else "protein_id"
    hits = query_index[query_index[key_col].astype(str) == str(query_protein_id)]
    if hits.empty:
        raise ValueError(f"query protein not found in query embedding index: {query_protein_id}")
    if len(hits) > 1:
        raise ValueError(f"duplicated query protein in query embedding index: {query_protein_id}")
    return int(hits.iloc[0]["embedding_row"])


def load_motif_cache(cache_dir: Path):
    index = pd.read_csv(cache_dir / "motif_feature_index.tsv", sep="\t")
    latent = np.load(cache_dir / "motif_latent.npy").astype(np.float32)
    subtype = np.load(cache_dir / "motif_subtype_logits.npy").astype(np.float32)
    seed = np.load(cache_dir / "motif_seed_logits.npy").astype(np.float32)
    all_features = np.concatenate([latent, subtype, seed], axis=1).astype(np.float32)
    train_rows = index[index["source"].astype(str) == "music_clip_train"]
    if train_rows.empty:
        raise ValueError("motif cache missing music_clip_train rows")
    train_pos = train_rows.index.to_numpy()
    train_features = all_features[train_pos]
    mean = train_features.mean(axis=0, keepdims=True)
    std = train_features.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return index, all_features, mean.astype(np.float32), std.astype(np.float32)


def resolve_external_motif_features(
    rbp_id: str,
    index: pd.DataFrame,
    all_features: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    hits = index[(index["source"].astype(str) == "external") & (index["protein_id"].astype(str) == rbp_id)]
    if hits.empty:
        raise ValueError(f"external motif features not found in cache for {rbp_id}")
    if len(hits) > 1:
        raise ValueError(f"duplicated external motif features in cache for {rbp_id}")
    row_pos = int(hits.index[0])
    feat = all_features[row_pos : row_pos + 1]
    feat = (feat - mean) / std
    feat = np.nan_to_num(feat, nan=0.0, posinf=10.0, neginf=-10.0).astype(np.float32)
    return feat


def load_checkpoint_bundle(checkpoint_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict):
        state = checkpoint.get("model_state_dict", checkpoint)
        stage = checkpoint.get("stage")
        args = checkpoint.get("args", {})
        binding_score_mode = checkpoint.get("binding_score_mode")
        use_joint_head = checkpoint.get("use_joint_head")
    else:
        state = checkpoint
        stage = None
        args = {}
        binding_score_mode = None
        use_joint_head = None
    return checkpoint, state, stage, args, binding_score_mode, use_joint_head


def build_model(cfg, checkpoint_path: Path, device: torch.device):
    model = OriginalLogicUnifiedRBPModel(cfg, load_pretrained=True).to(device)
    checkpoint, state, stage, args, binding_score_mode, use_joint_head = load_checkpoint_bundle(checkpoint_path)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, {
        "checkpoint": checkpoint,
        "stage": stage,
        "args": args,
        "binding_score_mode": binding_score_mode,
        "use_joint_head": use_joint_head,
    }


@torch.no_grad()
def build_motif_outputs(
    model: OriginalLogicUnifiedRBPModel,
    cfg,
    query_ids: Sequence[str],
    query_fasta: Path,
    kmers: np.ndarray,
    top_k: int,
    device: torch.device,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, np.ndarray]]:
    fasta_records = read_fasta(query_fasta)
    sequences = []
    phys = []
    labels = []
    full_ids = []
    for rbp_id in query_ids:
        query_protein_id = cfg.rbp_id_to_query_protein_id[rbp_id]
        seq = fasta_records.get(query_protein_id)
        if not seq:
            raise ValueError(f"protein sequence not found in query fasta: {query_protein_id}")
        seq = clean_protein(seq)
        sequences.append(seq)
        phys.append(phys_feats(seq))
        labels.append(kingdom_label_from_query_id(query_protein_id))
        full_ids.append(query_protein_id)
    outputs = model.forward_motif(
        sequences,
        torch.tensor(phys, dtype=torch.float32, device=device),
        torch.tensor(labels, dtype=torch.long, device=device),
    )
    reconstructed = outputs.get("rescued_z", outputs["reconstructed_z"]).detach().cpu().numpy().astype(np.float32)
    top_rows = []
    summary_rows = []
    motif_profiles = {}
    for rbp_id, query_protein_id, values in zip(query_ids, full_ids, reconstructed):
        motif_profiles[rbp_id] = values.astype(np.float32, copy=True)
        pos_idx = topk_indices_desc(values, top_k)
        neg_idx = topk_indices_asc(values, top_k)
        pos_kmers = []
        neg_kmers = []
        for rank, idx in enumerate(pos_idx, start=1):
            kmer = str(kmers[idx])
            score = float(values[idx])
            pos_kmers.append(kmer)
            top_rows.append(
                {
                    "rbp_id": rbp_id,
                    "query_protein_id": query_protein_id,
                    "direction": "positive",
                    "rank": rank,
                    "kmer": kmer,
                    "predicted_zscore": score,
                }
            )
        for rank, idx in enumerate(neg_idx, start=1):
            kmer = str(kmers[idx])
            score = float(values[idx])
            neg_kmers.append(kmer)
            top_rows.append(
                {
                    "rbp_id": rbp_id,
                    "query_protein_id": query_protein_id,
                    "direction": "negative",
                    "rank": rank,
                    "kmer": kmer,
                    "predicted_zscore": score,
                }
            )
        summary_rows.append(
            {
                "rbp_id": rbp_id,
                "query_protein_id": query_protein_id,
                "protein_length": len(fasta_records[query_protein_id]),
                "top_positive_kmers": ",".join(pos_kmers[:10]),
                "top_negative_kmers": ",".join(neg_kmers[:10]),
                "max_predicted_zscore": float(values[pos_idx[0]]),
                "min_predicted_zscore": float(values[neg_idx[0]]),
            }
        )
    return pd.DataFrame(top_rows), pd.DataFrame(summary_rows), motif_profiles


@torch.no_grad()
def score_windows_for_rbp(
    model: OriginalLogicUnifiedRBPModel,
    frame: pd.DataFrame,
    rna_len: int,
    query_embedding: np.ndarray,
    motif_features: np.ndarray,
    device: torch.device,
    batch_size: int,
    use_joint_head: bool,
) -> pd.DataFrame:
    query_embedding_tensor = torch.from_numpy(query_embedding.astype(np.float32)).to(device)
    if use_joint_head:
        motif_tensor = torch.from_numpy(motif_features.astype(np.float32)).to(device)
        projected_protein = model.binding_model.protein_projector(query_embedding_tensor)
        projected_motif = model.motif_binding_projector(motif_tensor)

    base_logits_all: List[np.ndarray] = []
    joint_logits_all: List[np.ndarray] = []
    for start in range(0, len(frame), batch_size):
        subset = frame.iloc[start : start + batch_size]
        rna_codes = encode_rna_sequences(subset["rna_seq"].tolist(), rna_len)
        rna_tensor = torch.from_numpy(rna_codes.astype(np.int64)).to(device)
        protein_batch = query_embedding_tensor.expand(len(subset), -1)
        base_logits = model.binding_model.forward_with_protein_vectors(rna_tensor, protein_batch)
        base_logits_all.append(base_logits.detach().cpu().numpy().astype(np.float32))
        if use_joint_head:
            rna_vec = model.binding_model.encode_rna(rna_tensor)
            joint_logits = model.motif_aware_classifier(
                torch.cat(
                    [
                        rna_vec,
                        projected_protein.expand(len(subset), -1),
                        projected_motif.expand(len(subset), -1),
                    ],
                    dim=1,
                )
            ).squeeze(1)
            joint_logits_all.append(joint_logits.detach().cpu().numpy().astype(np.float32))
    out = frame.copy()
    out["base_logit"] = np.concatenate(base_logits_all)
    out["base_score"] = 1.0 / (1.0 + np.exp(-out["base_logit"].to_numpy(dtype=np.float32)))
    if use_joint_head:
        out["joint_logit"] = np.concatenate(joint_logits_all)
        out["joint_score"] = 1.0 / (1.0 + np.exp(-out["joint_logit"].to_numpy(dtype=np.float32)))
        out["score_delta_joint_minus_base"] = out["joint_score"] - out["base_score"]
    else:
        out["joint_logit"] = np.nan
        out["joint_score"] = np.nan
        out["score_delta_joint_minus_base"] = np.nan
    return out


def build_kmer_index(kmers: np.ndarray) -> Dict[str, int]:
    return {str(kmer): idx for idx, kmer in enumerate(kmers.astype(str).tolist())}


def compute_window_motif_statistics(
    seq: str,
    motif_profile: np.ndarray,
    kmer_index: Dict[str, int],
    kmer_size: int,
    top_n: int,
) -> Tuple[float, float, float, float, int]:
    seq = str(seq).upper().replace("T", "U")
    if len(seq) < kmer_size:
        return 0.0, 0.0, 0.0, 0.0, 0
    scores: List[float] = []
    for start in range(len(seq) - kmer_size + 1):
        kmer = seq[start : start + kmer_size]
        idx = kmer_index.get(kmer)
        if idx is None:
            continue
        scores.append(float(motif_profile[idx]))
    if not scores:
        return 0.0, 0.0, 0.0, 0.0, 0
    arr = np.asarray(scores, dtype=np.float32)
    use_top_n = max(1, min(int(top_n), len(arr)))
    top_values = np.sort(arr)[-use_top_n:]
    positive_fraction = float((arr > 0).mean())
    return float(arr.max()), float(top_values.mean()), float(arr.mean()), positive_fraction, int(len(arr))


def robust_center_scale(raw: np.ndarray) -> Tuple[np.ndarray, float, float]:
    raw = np.asarray(raw, dtype=np.float32)
    center = float(np.median(raw))
    q75, q25 = np.percentile(raw, [75, 25])
    scale = float(q75 - q25)
    if not np.isfinite(scale) or scale < 1e-6:
        scale = float(np.std(raw))
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0
    score = (raw - center) / scale
    return score.astype(np.float32), center, scale


def compute_structure_statistics(
    structure_vectors: np.ndarray,
    high_threshold: float,
    low_threshold: float,
) -> Dict[str, np.ndarray]:
    arr = np.asarray(structure_vectors, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("structure vectors must be a 2D array")
    return {
        "paired_probability_mean": arr.mean(axis=1).astype(np.float32),
        "paired_probability_median": np.median(arr, axis=1).astype(np.float32),
        "paired_probability_max": arr.max(axis=1).astype(np.float32),
        "fraction_high_paired": (arr >= float(high_threshold)).mean(axis=1).astype(np.float32),
        "fraction_low_paired": (arr <= float(low_threshold)).mean(axis=1).astype(np.float32),
    }


def compute_structure_raw_signal(
    stats: Dict[str, np.ndarray],
    score_mode: str,
) -> np.ndarray:
    if score_mode == "mean":
        return stats["paired_probability_mean"]
    if score_mode == "fraction_high":
        return stats["fraction_high_paired"]
    if score_mode == "max":
        return stats["paired_probability_max"]
    if score_mode != "combined":
        raise ValueError(f"unsupported structure score mode: {score_mode}")
    raw = (
        0.45 * stats["paired_probability_mean"]
        + 0.15 * stats["paired_probability_median"]
        + 0.20 * stats["paired_probability_max"]
        + 0.20 * stats["fraction_high_paired"]
    )
    return raw.astype(np.float32)


def add_posthoc_motif_scores(
    frame: pd.DataFrame,
    motif_profile: np.ndarray,
    kmers: np.ndarray,
    motif_alpha: float,
    motif_top_n: int,
    structure_vectors: np.ndarray | None = None,
    structure_alpha: float = 0.0,
    structure_score_mode: str = "combined",
    structure_high_threshold: float = 0.7,
    structure_low_threshold: float = 0.3,
) -> pd.DataFrame:
    kmer_size = len(str(kmers[0]))
    kmer_index = build_kmer_index(kmers)
    stats = [
        compute_window_motif_statistics(seq, motif_profile, kmer_index, kmer_size, motif_top_n)
        for seq in frame["rna_seq"].tolist()
    ]
    stat_array = np.asarray(stats, dtype=np.float32)
    out = frame.copy()
    out["motif_match_max_zscore"] = stat_array[:, 0]
    out["motif_match_top_mean_zscore"] = stat_array[:, 1]
    out["motif_match_mean_zscore"] = stat_array[:, 2]
    out["motif_positive_fraction"] = stat_array[:, 3]
    out["motif_scanned_kmer_count"] = stat_array[:, 4].astype(np.int32)
    motif_match_score, motif_center, motif_scale = robust_center_scale(
        out["motif_match_top_mean_zscore"].to_numpy(dtype=np.float32)
    )
    out["motif_match_score"] = motif_match_score.astype(np.float32)
    out["motif_match_center"] = motif_center
    out["motif_match_scale"] = motif_scale
    posthoc_logit = out["base_logit"].to_numpy(dtype=np.float32) + float(motif_alpha) * out["motif_match_score"].to_numpy(dtype=np.float32)
    if structure_vectors is not None:
        structure_stats = compute_structure_statistics(
            structure_vectors=structure_vectors,
            high_threshold=structure_high_threshold,
            low_threshold=structure_low_threshold,
        )
        for column, values in structure_stats.items():
            out[column] = values
        structure_raw = compute_structure_raw_signal(structure_stats, structure_score_mode)
        structure_score, structure_center, structure_scale = robust_center_scale(structure_raw)
        out["structure_match_raw"] = structure_raw.astype(np.float32)
        out["structure_match_score"] = structure_score.astype(np.float32)
        out["structure_match_center"] = structure_center
        out["structure_match_scale"] = structure_scale
        posthoc_logit = posthoc_logit + float(structure_alpha) * structure_score
    else:
        out["paired_probability_mean"] = np.nan
        out["paired_probability_median"] = np.nan
        out["paired_probability_max"] = np.nan
        out["fraction_high_paired"] = np.nan
        out["fraction_low_paired"] = np.nan
        out["structure_match_raw"] = np.nan
        out["structure_match_score"] = np.nan
        out["structure_match_center"] = np.nan
        out["structure_match_scale"] = np.nan
    out["posthoc_logit"] = posthoc_logit
    out["posthoc_score"] = 1.0 / (1.0 + np.exp(-out["posthoc_logit"].to_numpy(dtype=np.float32)))
    out["score_delta_posthoc_minus_base"] = out["posthoc_score"] - out["base_score"]
    return out


def finalize_gene_rows(rbp_id: str, frame: pd.DataFrame, score_col: str) -> pd.DataFrame:
    rows = []
    for gene_id, group in frame.groupby("gene_id", sort=False):
        g = group.sort_values(score_col, ascending=False)
        top_scores = g[score_col].head(5).to_numpy(dtype=float)
        best = g.iloc[0]
        rows.append(
            {
                "rbp_id": rbp_id,
                "gene_id": str(gene_id),
                "transcript_id": str(best["transcript_id"]),
                "max_score": float(best[score_col]),
                "mean_top3_score": float(np.mean(top_scores[:3])),
                "mean_top5_score": float(np.mean(top_scores[:5])),
                "best_window_start": int(best["window_start"]),
                "best_window_end": int(best["window_end"]),
                "best_window_seq": str(best["rna_seq"]),
                "ranking_score_type": score_col,
                "best_window_logit": float(
                    best["joint_logit"]
                    if score_col == "joint_score"
                    else best["posthoc_logit"]
                    if score_col == "posthoc_score"
                    else best["base_logit"]
                ),
                "best_window_base_score": float(best["base_score"]),
                "best_window_joint_score": float(best["joint_score"]) if pd.notna(best["joint_score"]) else float("nan"),
                "best_window_posthoc_score": float(best["posthoc_score"]) if "posthoc_score" in best.index and pd.notna(best["posthoc_score"]) else float("nan"),
                "best_window_motif_match_score": float(best["motif_match_score"]) if "motif_match_score" in best.index and pd.notna(best["motif_match_score"]) else float("nan"),
                "best_window_structure_match_score": float(best["structure_match_score"]) if "structure_match_score" in best.index and pd.notna(best["structure_match_score"]) else float("nan"),
                "best_window_paired_probability_mean": float(best["paired_probability_mean"]) if "paired_probability_mean" in best.index and pd.notna(best["paired_probability_mean"]) else float("nan"),
                "best_window_fraction_high_paired": float(best["fraction_high_paired"]) if "fraction_high_paired" in best.index and pd.notna(best["fraction_high_paired"]) else float("nan"),
                "n_windows": int(len(g)),
            }
        )
    gene_scores = pd.DataFrame(rows)
    if gene_scores.empty:
        return gene_scores
    gene_scores = gene_scores.sort_values(["max_score", "gene_id"], ascending=[False, True]).reset_index(drop=True)
    gene_scores["rank"] = np.arange(1, len(gene_scores) + 1)
    return gene_scores


def compute_binary_metrics(labels: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    labels = labels.astype(int)
    scores = scores.astype(float)
    if len(np.unique(labels)) < 2:
        return {"auc": float("nan"), "auprc": float("nan"), "positive_median": float("nan"), "negative_median": float("nan")}
    pos_scores = scores[labels == 1]
    neg_scores = scores[labels == 0]
    return {
        "auc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "positive_median": float(np.median(pos_scores)) if len(pos_scores) else float("nan"),
        "negative_median": float(np.median(neg_scores)) if len(neg_scores) else float("nan"),
    }


def summarize_labeled_predictions(rbp_id: str, frame: pd.DataFrame) -> Dict[str, float | str]:
    labels = frame["label"].to_numpy(dtype=int)
    base_metrics = compute_binary_metrics(labels, frame["base_score"].to_numpy(dtype=float))
    row = {
        "rbp_id": rbp_id,
        "n_rows": int(len(frame)),
        "n_positive": int((labels == 1).sum()),
        "n_negative": int((labels == 0).sum()),
        "base_auc": base_metrics["auc"],
        "base_auprc": base_metrics["auprc"],
        "base_positive_median": base_metrics["positive_median"],
        "base_negative_median": base_metrics["negative_median"],
    }
    if frame["joint_score"].notna().any():
        joint_metrics = compute_binary_metrics(labels, frame["joint_score"].to_numpy(dtype=float))
        row.update(
            {
                "joint_auc": joint_metrics["auc"],
                "joint_auprc": joint_metrics["auprc"],
                "joint_positive_median": joint_metrics["positive_median"],
                "joint_negative_median": joint_metrics["negative_median"],
            }
        )
    else:
        row.update(
            {
                "joint_auc": float("nan"),
                "joint_auprc": float("nan"),
                "joint_positive_median": float("nan"),
                "joint_negative_median": float("nan"),
            }
        )
    if "posthoc_score" in frame.columns and frame["posthoc_score"].notna().any():
        posthoc_metrics = compute_binary_metrics(labels, frame["posthoc_score"].to_numpy(dtype=float))
        row.update(
            {
                "posthoc_auc": posthoc_metrics["auc"],
                "posthoc_auprc": posthoc_metrics["auprc"],
                "posthoc_positive_median": posthoc_metrics["positive_median"],
                "posthoc_negative_median": posthoc_metrics["negative_median"],
            }
        )
    else:
        row.update(
            {
                "posthoc_auc": float("nan"),
                "posthoc_auprc": float("nan"),
                "posthoc_positive_median": float("nan"),
                "posthoc_negative_median": float("nan"),
            }
        )
    return row


def write_top_gene_lists(out_dir: Path, rbp_id: str, gene_scores: pd.DataFrame):
    lists_dir = out_dir / "top_gene_lists"
    lists_dir.mkdir(parents=True, exist_ok=True)
    safe_rbp = str(rbp_id).replace("/", "_")
    for k in (200, 500, 1000):
        subset = gene_scores.head(k)
        subset[["gene_id"]].to_csv(
            lists_dir / f"{safe_rbp}_Top{k}_genes.txt",
            sep="\t",
            index=False,
            header=False,
        )


def parse_args():
    cfg = get_config()
    parser = argparse.ArgumentParser(
        description="Predict motif preference and motif-aware RNA-window binding for external RBPs."
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/unified_original_logic_model_binding.pt",
        help="Unified checkpoint to use in single-checkpoint mode.",
    )
    parser.add_argument(
        "--motif-checkpoint",
        default=None,
        help="Motif checkpoint used for prediction-stage motif assistance.",
    )
    parser.add_argument(
        "--binding-checkpoint",
        default=None,
        help="Binding checkpoint used for prediction-stage motif assistance.",
    )
    parser.add_argument(
        "--prediction-mode",
        choices=["auto", "single", "posthoc"],
        default="auto",
        help="single: one checkpoint only; posthoc: motif and binding checkpoints are used separately.",
    )
    parser.add_argument(
        "--window-tsv",
        default=None,
        help="Prepared RNA window table. Required columns: gene_id, transcript_id, window_start, window_end, rna_seq.",
    )
    parser.add_argument(
        "--query-rbp-ids",
        default="AtGRP7,AtGRP8,LOC_Os05g24160.1",
        help="Comma-separated short RBP IDs from unified_config.rbp_id_to_query_protein_id.",
    )
    parser.add_argument(
        "--query-fasta",
        default=(
            Path(cfg.binding_project_dir)
            / "08_embeddings/plant_rbp_extended_query_fasta/rice7_plus_validated_plant_rbp.fasta"
        ),
    )
    parser.add_argument(
        "--query-embedding-npy",
        default=(
            Path(cfg.binding_project_dir)
            / "08_embeddings/plant_rbp_extended_query_fasta/esm2_t33_650M_full_length/protein_embeddings.npy"
        ),
    )
    parser.add_argument(
        "--query-embedding-index",
        default=(
            Path(cfg.binding_project_dir)
            / "08_embeddings/plant_rbp_extended_query_fasta/esm2_t33_650M_full_length/protein_embedding_index.tsv"
        ),
    )
    parser.add_argument(
        "--motif-cache-dir",
        default="data/original_logic_cache",
        help="Motif cache generated by cache_original_logic_motif_features.py.",
    )
    parser.add_argument(
        "--motif-profile-npz",
        default="data/multitask/motif_profiles.npz",
        help="NPZ containing the k-mer order used by motif reconstruction.",
    )
    parser.add_argument(
        "--override-motif-profile-csv",
        default=None,
        help="Optional external motif profile matrix CSV/TSV used only when explicitly provided.",
    )
    parser.add_argument(
        "--motif-profile-mode",
        choices=["auto_teacher", "direct", "override"],
        default="auto_teacher",
        help="How to build the motif profile used for motif output and posthoc RNA-window scoring.",
    )
    parser.add_argument("--auto-teacher-domain-boundary-csv", default=None)
    parser.add_argument("--auto-teacher-query-kingdom-label", type=int, default=2)
    parser.add_argument("--auto-teacher-window-size", type=int, default=256)
    parser.add_argument("--auto-teacher-stride", type=int, default=64)
    parser.add_argument("--auto-teacher-min-window", type=int, default=128)
    parser.add_argument("--auto-teacher-top-neighbors", type=int, default=12)
    parser.add_argument("--auto-teacher-select-top-windows", type=int, default=1)
    parser.add_argument("--auto-teacher-min-confidence", type=float, default=0.25)
    parser.add_argument("--auto-teacher-disable-neighbor-prior", action="store_true")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--top-kmers", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--motif-alpha", type=float, default=1.0)
    parser.add_argument("--motif-top-n", type=int, default=3)
    parser.add_argument("--structure-features-npy", default=None)
    parser.add_argument("--structure-meta-json", default=None)
    parser.add_argument("--structure-alpha", type=float, default=0.0)
    parser.add_argument(
        "--structure-score-mode",
        choices=["combined", "mean", "fraction_high", "max"],
        default="combined",
    )
    parser.add_argument("--structure-high-threshold", type=float, default=0.7)
    parser.add_argument("--structure-low-threshold", type=float, default=0.3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--allow-untrained-joint-head",
        action="store_true",
        help="Allow joint-score output even for a motif-stage checkpoint. Normally motif-stage only reports base_score.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = get_config()
    work_dir = Path(cfg.work_dir)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = work_dir / checkpoint_path
    motif_checkpoint_path = Path(args.motif_checkpoint) if args.motif_checkpoint else None
    if motif_checkpoint_path is not None and not motif_checkpoint_path.is_absolute():
        motif_checkpoint_path = work_dir / motif_checkpoint_path
    binding_checkpoint_path = Path(args.binding_checkpoint) if args.binding_checkpoint else None
    if binding_checkpoint_path is not None and not binding_checkpoint_path.is_absolute():
        binding_checkpoint_path = work_dir / binding_checkpoint_path
    motif_cache_dir = Path(args.motif_cache_dir)
    if not motif_cache_dir.is_absolute():
        motif_cache_dir = work_dir / motif_cache_dir
    motif_profile_npz = Path(args.motif_profile_npz)
    if not motif_profile_npz.is_absolute():
        motif_profile_npz = work_dir / motif_profile_npz
    override_motif_profile_csv = Path(args.override_motif_profile_csv) if args.override_motif_profile_csv else None
    if override_motif_profile_csv is not None and not override_motif_profile_csv.is_absolute():
        override_motif_profile_csv = work_dir / override_motif_profile_csv
    auto_teacher_domain_boundary_csv = Path(args.auto_teacher_domain_boundary_csv) if args.auto_teacher_domain_boundary_csv else None
    if auto_teacher_domain_boundary_csv is not None and not auto_teacher_domain_boundary_csv.is_absolute():
        auto_teacher_domain_boundary_csv = work_dir / auto_teacher_domain_boundary_csv
    structure_features_npy = Path(args.structure_features_npy) if args.structure_features_npy else None
    if structure_features_npy is not None and not structure_features_npy.is_absolute():
        structure_features_npy = work_dir / structure_features_npy
    structure_meta_json = Path(args.structure_meta_json) if args.structure_meta_json else None
    if structure_meta_json is not None and not structure_meta_json.is_absolute():
        structure_meta_json = work_dir / structure_meta_json
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = work_dir / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    query_ids = parse_query_ids(args.query_rbp_ids, cfg)
    prediction_mode = args.prediction_mode
    if prediction_mode == "auto":
        prediction_mode = "posthoc" if motif_checkpoint_path or binding_checkpoint_path else "single"

    if prediction_mode == "posthoc":
        if motif_checkpoint_path is None or binding_checkpoint_path is None:
            raise ValueError("posthoc mode requires both --motif-checkpoint and --binding-checkpoint")
        motif_model, motif_checkpoint_meta = build_model(cfg, motif_checkpoint_path, device)
        binding_model, binding_checkpoint_meta = build_model(cfg, binding_checkpoint_path, device)
        checkpoint_stage = binding_checkpoint_meta.get("stage")
        checkpoint_binding_score_mode = "base"
        use_joint_head = False
    else:
        model, checkpoint_meta = build_model(cfg, checkpoint_path, device)
        motif_model = model
        binding_model = model
        motif_checkpoint_meta = checkpoint_meta
        binding_checkpoint_meta = checkpoint_meta
        checkpoint_stage = checkpoint_meta.get("stage")
        checkpoint_use_joint_head = checkpoint_meta.get("use_joint_head")
        checkpoint_binding_score_mode = checkpoint_meta.get("binding_score_mode")
        if checkpoint_use_joint_head is not None:
            use_joint_head = bool(checkpoint_use_joint_head)
        elif checkpoint_binding_score_mode is not None:
            use_joint_head = checkpoint_binding_score_mode == "joint"
        else:
            use_joint_head = checkpoint_stage in {"late_fusion", "joint"} or args.allow_untrained_joint_head

    kmers = np.load(motif_profile_npz)["kmers"].astype(str)
    if args.motif_profile_mode == "override":
        if override_motif_profile_csv is None:
            raise ValueError("motif_profile_mode=override requires --override-motif-profile-csv")
        motif_top, motif_summary, motif_profiles = load_override_motif_profiles(
            profile_csv=override_motif_profile_csv,
            kmers=kmers,
            query_ids=query_ids,
            cfg=cfg,
            top_k=args.top_kmers,
        )
    elif args.motif_profile_mode == "auto_teacher":
        motif_profiles, teacher_summary, teacher_selected, teacher_conflicts = build_auto_teacher_outputs(
            model=motif_model,
            cfg=cfg,
            query_ids=query_ids,
            query_fasta=Path(args.query_fasta),
            motif_profile_npz=motif_profile_npz,
            device=device,
            out_dir=out_dir,
            domain_boundary_csv=auto_teacher_domain_boundary_csv,
            query_kingdom_label=args.auto_teacher_query_kingdom_label,
            window_size=args.auto_teacher_window_size,
            stride=args.auto_teacher_stride,
            min_window=args.auto_teacher_min_window,
            top_neighbors=args.auto_teacher_top_neighbors,
            select_top_windows=args.auto_teacher_select_top_windows,
            min_confidence=args.auto_teacher_min_confidence,
            disable_neighbor_prior=args.auto_teacher_disable_neighbor_prior,
            batch_size=max(1, min(16, args.batch_size if args.batch_size > 0 else 8)),
        )
        motif_top, motif_summary = summarize_profile_dict(
            profile_map=motif_profiles,
            cfg=cfg,
            kmers=kmers,
            top_k=args.top_kmers,
            extra_summary=teacher_summary,
        )
    else:
        motif_top, motif_summary, motif_profiles = build_motif_outputs(
            model=motif_model,
            cfg=cfg,
            query_ids=query_ids,
            query_fasta=Path(args.query_fasta),
            kmers=kmers,
            top_k=args.top_kmers,
            device=device,
        )
    motif_top.to_csv(out_dir / "motif_top_kmers.tsv", sep="\t", index=False)
    motif_summary.to_csv(out_dir / "motif_summary.tsv", sep="\t", index=False)

    report: Dict[str, object] = {
        "prediction_mode": prediction_mode,
        "checkpoint": str(checkpoint_path) if prediction_mode == "single" else None,
        "motif_checkpoint": str(motif_checkpoint_path) if motif_checkpoint_path is not None else str(checkpoint_path),
        "binding_checkpoint": str(binding_checkpoint_path) if binding_checkpoint_path is not None else str(checkpoint_path),
        "checkpoint_stage": checkpoint_stage,
        "checkpoint_binding_score_mode": checkpoint_binding_score_mode,
        "device": str(device),
        "query_rbp_ids": query_ids,
        "use_joint_head": bool(use_joint_head),
        "motif_alpha": float(args.motif_alpha),
        "motif_top_n": int(args.motif_top_n),
        "structure_alpha": float(args.structure_alpha),
        "structure_score_mode": args.structure_score_mode,
        "structure_features_npy": str(structure_features_npy) if structure_features_npy is not None else None,
        "structure_meta_json": str(structure_meta_json) if structure_meta_json is not None else None,
        "motif_profile_mode": args.motif_profile_mode,
        "override_motif_profile_csv": str(override_motif_profile_csv) if override_motif_profile_csv is not None else None,
        "motif_top_kmers_tsv": str(out_dir / "motif_top_kmers.tsv"),
        "motif_summary_tsv": str(out_dir / "motif_summary.tsv"),
    }
    if args.motif_profile_mode == "auto_teacher":
        report["auto_teacher_fused_z_matrix_csv"] = str(out_dir / "auto_teacher_fused_z_matrix.csv")
        report["auto_teacher_selected_windows_tsv"] = str(out_dir / "auto_teacher_selected_windows.csv")
        report["auto_teacher_conflicts_tsv"] = str(out_dir / "auto_teacher_conflicts.tsv")
        report["auto_teacher_summary_tsv"] = str(out_dir / "auto_teacher_summary.tsv")
        report["auto_teacher_conflict_candidate_z_matrix_csv"] = str(out_dir / "auto_teacher_conflict_candidate_z_matrix.csv")
        report["auto_teacher_conflict_candidate_windows_tsv"] = str(out_dir / "auto_teacher_conflict_candidate_windows.csv")

    if args.window_tsv:
        window_path = Path(args.window_tsv)
        if not window_path.is_absolute():
            window_path = work_dir / window_path
        frame = load_window_table(window_path)
        structure_all = None
        structure_meta = None
        if structure_features_npy is not None or structure_meta_json is not None:
            if structure_features_npy is None or structure_meta_json is None:
                raise ValueError("use both --structure-features-npy and --structure-meta-json together")
            structure_all, structure_meta = load_structure_cache_for_frame(
                frame=frame,
                structure_npy=structure_features_npy,
                structure_meta_json=structure_meta_json,
            )
            report["resolved_structure_rows"] = int(structure_all.shape[0])
            report["resolved_structure_rna_len"] = int(structure_all.shape[1])
            report["resolved_structure_method"] = structure_meta.get("method")
        query_embeddings, query_index = load_query_embedding_table(
            Path(args.query_embedding_npy),
            Path(args.query_embedding_index),
            Path(cfg.binding_train_embedding_npy),
        )
        motif_index, motif_all, motif_mean, motif_std = load_motif_cache(motif_cache_dir)

        all_window_rows = []
        all_gene_rows = []
        eval_rows = []
        for rbp_id in query_ids:
            query_protein_id = cfg.rbp_id_to_query_protein_id[rbp_id]
            query_row = resolve_query_embedding_row(query_index, query_protein_id)
            query_embedding = query_embeddings[query_row : query_row + 1]
            motif_features = resolve_external_motif_features(rbp_id, motif_index, motif_all, motif_mean, motif_std)
            if "rbp_id" in frame.columns:
                mask = frame["rbp_id"].astype(str).isin([rbp_id, query_protein_id]).to_numpy()
                subset = frame.loc[mask].copy()
                structure_subset = np.asarray(structure_all[mask], dtype=np.float32) if structure_all is not None else None
            else:
                subset = frame.copy()
                subset["rbp_id"] = rbp_id
                structure_subset = np.asarray(structure_all, dtype=np.float32) if structure_all is not None else None
            if subset.empty:
                continue
            scored = score_windows_for_rbp(
                model=binding_model,
                frame=subset.reset_index(drop=True),
                rna_len=cfg.rna_len,
                query_embedding=query_embedding,
                motif_features=motif_features,
                device=device,
                batch_size=args.batch_size,
                use_joint_head=use_joint_head,
            )
            apply_posthoc = (
                prediction_mode == "posthoc"
                or args.motif_profile_mode in {"auto_teacher", "override"}
                or structure_subset is not None
            )
            if apply_posthoc:
                scored = add_posthoc_motif_scores(
                    frame=scored,
                    motif_profile=motif_profiles[rbp_id],
                    kmers=kmers,
                    motif_alpha=args.motif_alpha,
                    motif_top_n=args.motif_top_n,
                    structure_vectors=structure_subset,
                    structure_alpha=args.structure_alpha,
                    structure_score_mode=args.structure_score_mode,
                    structure_high_threshold=args.structure_high_threshold,
                    structure_low_threshold=args.structure_low_threshold,
                )
            scored["scorer_rbp_id"] = rbp_id
            scored["query_protein_id"] = query_protein_id
            scored["checkpoint_stage"] = checkpoint_stage
            scored["prediction_mode"] = prediction_mode
            all_window_rows.append(scored)
            if apply_posthoc:
                ranking_score_col = "posthoc_score"
            else:
                ranking_score_col = "joint_score" if use_joint_head else "base_score"
            gene_scores = finalize_gene_rows(rbp_id, scored, ranking_score_col)
            if not gene_scores.empty:
                all_gene_rows.append(gene_scores)
                write_top_gene_lists(out_dir, rbp_id, gene_scores)
            if "label" in scored.columns:
                eval_rows.append(summarize_labeled_predictions(rbp_id, scored))

        if all_window_rows:
            window_scores = pd.concat(all_window_rows, ignore_index=True)
            window_scores.to_csv(out_dir / "window_scores.tsv.gz", sep="\t", index=False, compression="gzip")
            report["window_scores_tsv_gz"] = str(out_dir / "window_scores.tsv.gz")
        else:
            window_scores = pd.DataFrame()
        if all_gene_rows:
            gene_scores = pd.concat(all_gene_rows, ignore_index=True)
            gene_scores.to_csv(out_dir / "gene_scores.tsv", sep="\t", index=False)
            report["gene_scores_tsv"] = str(out_dir / "gene_scores.tsv")
        if eval_rows:
            eval_df = pd.DataFrame(eval_rows)
            eval_df.to_csv(out_dir / "label_eval_summary.tsv", sep="\t", index=False)
            report["label_eval_summary_tsv"] = str(out_dir / "label_eval_summary.tsv")
        report["window_input_tsv"] = str(window_path)

    with open(out_dir / "prediction_report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
