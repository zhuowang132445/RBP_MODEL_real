#!/usr/bin/env python3
"""V6-faithful motif teacher generation for unified prediction."""

from __future__ import annotations

import importlib.util
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def load_module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_v6_window_scan_module(cfg):
    script_path = Path(cfg.motif_snapshot_dir) / "scripts/esm_v2_window_scan.py"
    if not script_path.exists():
        raise FileNotFoundError(f"V6 window scan script not found: {script_path}")
    script_dir = str(script_path.parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    return load_module_from_path("unified_v6_window_scan", script_path)


def resolve_snapshot_data_paths(cfg) -> Dict[str, Path]:
    data_dir = Path(cfg.motif_snapshot_dir) / "data"
    return {
        "train_fasta": data_dir / "seq_train.fasta",
        "train_matrix": data_dir / "zscore_train.tsv",
        "metadata_path": data_dir / "TableS1.xlsx",
        "metadata_tsv": data_dir / "rnacompete_metadata_eupri.tsv",
        "domain_map": data_dir / "domain_map.csv",
        "default_domain_boundary_csv": data_dir / "domain_boundaries/rice_rbps.domain_boundaries.csv",
    }


def normalize_kmer_order(kmers: Sequence[str]) -> List[str]:
    return [str(k).strip().upper().replace("T", "U") for k in kmers]


def batch_encode_unified(
    model,
    sequences: Sequence[str],
    phys_features: torch.Tensor,
    kingdom_labels: torch.Tensor,
    batch_size: int,
    device: torch.device,
    use_rescued_prediction: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    preds = []
    latents = []
    with torch.no_grad():
        for start in range(0, len(sequences), max(1, int(batch_size))):
            end = min(start + max(1, int(batch_size)), len(sequences))
            out = model.forward_motif(
                list(sequences[start:end]),
                phys_features[start:end].to(device),
                kingdom_labels[start:end].to(device),
            )
            pred = out["rescued_z"] if use_rescued_prediction else out["reconstructed_z"]
            preds.append(pred.detach().cpu())
            latents.append(out["motif_latent"].detach().cpu())
    return torch.cat(preds, dim=0), torch.cat(latents, dim=0)


def align_profile_map(profile_map: Dict[str, np.ndarray], source_kmers: Sequence[str], target_kmers: Sequence[str]) -> Dict[str, np.ndarray]:
    src = normalize_kmer_order(source_kmers)
    dst = normalize_kmer_order(target_kmers)
    if src == dst:
        return {key: np.asarray(value, dtype=np.float32) for key, value in profile_map.items()}
    index = {k: i for i, k in enumerate(src)}
    order = [index[k] for k in dst]
    return {
        key: np.asarray(value, dtype=np.float32)[order]
        for key, value in profile_map.items()
    }


def default_domain_boundary_csv(cfg) -> Path | None:
    path = resolve_snapshot_data_paths(cfg)["default_domain_boundary_csv"]
    return path if path.exists() else None


KINGDOM_LABEL_TO_NAME = {
    0: "animal",
    1: "plant",
    2: "fungi",
    3: "other",
}


@torch.no_grad()
def build_auto_teacher_outputs(
    model,
    cfg,
    query_ids: Sequence[str],
    query_fasta: Path,
    motif_profile_npz: Path,
    device: torch.device,
    out_dir: Path | None = None,
    domain_boundary_csv: Path | None = None,
    batch_size: int = 8,
    include_full_length: bool = True,
    window_size: int = 256,
    stride: int = 64,
    min_window: int = 128,
    top_neighbors: int = 12,
    select_top_windows: int = 1,
    min_confidence: float = 0.25,
    signal_top_k: int = 100,
    prefer_domains: str = "NTF2,RRM,KH",
    domain_padding: int = 30,
    domain_window_max_len: int = 320,
    domain_window_only: bool = False,
    domain_filter_mode: str = "prefer",
    disable_neighbor_prior: bool = False,
    prior_top_neighbors: int = 8,
    prior_alpha_max: float = 0.35,
    prior_corr_floor: float = 0.10,
    prior_min_cohesion: float = 0.05,
    motif_group_prior_alpha_max: float = 0.80,
    motif_group_prior_top_neighbors: int = 8,
    motif_group_prior_min_sim: float = 0.0,
    motif_group_prior_min_score: float = 0.80,
    motif_group_prior_max_opposite: float = 0.20,
    motif_group_prior_min_cohesion: float = -0.50,
    motif_group_prior_nonconflict_alpha_max: float | None = 0.20,
    motif_group_prior_template_alpha_max: float | None = 0.05,
    template_prior_signature_top_ids: int = 4,
    template_collapse_min_shared_prior: int = 2,
    uc_repeat_prior_only: bool = True,
    uc_repeat_label_topk: int = 100,
    uc_repeat_min_score: float = 3.0,
    uc_repeat_min_hits: int = 5,
    uc_repeat_max_a_frac: float = 0.15,
    uc_repeat_max_g_frac: float = 0.15,
    uc_repeat_min_transition: float = 1.8,
    export_conflict_candidates: bool = True,
    candidate_top_windows: int = 1,
    motif_group_topn: int = 20,
    motif_group_label_topk: int = 100,
    motif_group_min_score: float = 1.0,
    motif_group_margin: float = 0.25,
    motif_conflict_min_group_score: float = 0.45,
    motif_conflict_min_confidence: float = 0.25,
    motif_conflict_mode: str = "flag",
    motif_conflict_penalty: float = 0.35,
    conflict_domain_boundary_bonus: float = 0.20,
    query_kingdom_label: int = 1,
    query_kingdom_labels: Sequence[int] | None = None,
    use_rescued_prediction: bool = False,
):
    mod = load_v6_window_scan_module(cfg)
    data_paths = resolve_snapshot_data_paths(cfg)
    query_fasta = Path(query_fasta)
    motif_profile_npz = Path(motif_profile_npz)
    domain_boundary_csv = Path(domain_boundary_csv) if domain_boundary_csv else default_domain_boundary_csv(cfg)
    domain_map = mod.load_domain_map(str(data_paths["domain_map"]))
    prefer_domain_tokens = {item.upper() for item in mod.parse_csv_list(prefer_domains)}
    domain_boundaries = mod.load_domain_boundaries(str(domain_boundary_csv)) if domain_boundary_csv else {}
    protein_map = mod.read_fasta(str(query_fasta))
    full_lengths = {protein_id: len(mod.clean_seq(seq)) for protein_id, seq in protein_map.items()}

    windows: List[object] = []
    window_kingdom_labels: List[int] = []
    per_query_kingdom_labels = list(query_kingdom_labels) if query_kingdom_labels is not None else None
    if per_query_kingdom_labels is not None and len(per_query_kingdom_labels) != len(query_ids):
        raise ValueError("query_kingdom_labels length must match query_ids length")
    protein_kingdom_lookup: Dict[str, int] = {}
    for rbp_idx, rbp_id in enumerate(query_ids):
        query_protein_id = cfg.rbp_id_to_query_protein_id[rbp_id]
        seq = protein_map.get(query_protein_id) or protein_map.get(rbp_id)
        if not seq:
            raise ValueError(f"protein sequence not found in query fasta: {query_protein_id}")
        resolved_query_kingdom_label = (
            int(per_query_kingdom_labels[rbp_idx])
            if per_query_kingdom_labels is not None
            else int(query_kingdom_label)
        )
        protein_kingdom_lookup[rbp_id] = resolved_query_kingdom_label
        boundary_items = []
        boundary_items.extend(domain_boundaries.get(rbp_id, []))
        boundary_items.extend(domain_boundaries.get(query_protein_id, []))
        domain_windows = mod.build_domain_boundary_windows(
            protein_id=rbp_id,
            seq=seq,
            boundaries=boundary_items,
            prefer_types=prefer_domain_tokens,
            padding=domain_padding,
            min_window=min_window,
            max_window=domain_window_max_len,
        )
        windows.extend(domain_windows)
        window_kingdom_labels.extend([resolved_query_kingdom_label] * len(domain_windows))
        if not domain_window_only or not domain_windows:
            sliding_windows = mod.build_windows(
                protein_id=rbp_id,
                seq=seq,
                window_size=window_size,
                stride=stride,
                min_window=min_window,
                include_full_length=include_full_length,
            )
            windows.extend(sliding_windows)
            window_kingdom_labels.extend([resolved_query_kingdom_label] * len(sliding_windows))
    if not windows:
        raise ValueError("no windows generated for auto teacher")

    dataset = mod.RBPDataset(
        fasta_path=str(data_paths["train_fasta"]),
        matrix_path=str(data_paths["train_matrix"]),
        metadata_path=str(data_paths["metadata_path"]),
        metadata_id_col="RNAcompete experiment used for JPLE",
        metadata_cluster_col="Motif similarity cluster",
        metadata_kingdom_col="Kingdom",
        use_metadata_cluster=True,
        use_metadata_kingdom=True,
    )
    kmers_train = normalize_kmer_order(dataset.kmers)
    kmers_target = normalize_kmer_order(np.load(motif_profile_npz)["kmers"].astype(str))

    train_seqs = dataset.sequences
    train_phys = torch.tensor(np.asarray(dataset.phys_features), dtype=torch.float32)
    train_king = torch.tensor(np.asarray(dataset.kingdom_labels), dtype=torch.long)
    train_ids = np.asarray(dataset.experiment_ids)
    train_labels = np.nan_to_num(np.asarray(dataset.labels, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)

    query_seqs = [win.sequence for win in windows]
    query_phys = torch.tensor([mod.get_physchem_features(seq) for seq in query_seqs], dtype=torch.float32)
    query_king = torch.tensor(np.asarray(window_kingdom_labels, dtype=np.int64), dtype=torch.long)

    _, train_latents = batch_encode_unified(
        model=model,
        sequences=train_seqs,
        phys_features=train_phys,
        kingdom_labels=train_king,
        batch_size=batch_size,
        device=device,
        use_rescued_prediction=use_rescued_prediction,
    )
    query_preds, query_latents = batch_encode_unified(
        model=model,
        sequences=query_seqs,
        phys_features=query_phys,
        kingdom_labels=query_king,
        batch_size=batch_size,
        device=device,
        use_rescued_prediction=use_rescued_prediction,
    )

    train_motif_groups = mod.classify_training_motif_groups(
        labels=train_labels,
        kmers=kmers_train,
        train_ids=train_ids,
        topk=motif_group_label_topk,
        min_score=motif_group_min_score,
        margin=motif_group_margin,
    )
    uc_repeat_train_ids = mod.classify_uc_repeat_training_profiles(
        labels=train_labels,
        kmers=kmers_train,
        train_ids=train_ids,
        topk=uc_repeat_label_topk,
        min_score=uc_repeat_min_score,
        min_hits=uc_repeat_min_hits,
        max_a_frac=uc_repeat_max_a_frac,
        max_g_frac=uc_repeat_max_g_frac,
        min_transition=uc_repeat_min_transition,
    )

    all_window_predictions: Dict[str, List[object]] = defaultdict(list)
    for win_idx, win in enumerate(windows):
        pred_vec = query_preds[win_idx].numpy()
        latent_vec = query_latents[win_idx]
        sim = F.cosine_similarity(latent_vec.unsqueeze(0), train_latents, dim=1).numpy()
        order = np.argsort(-sim)
        confidence, confidence_components = mod.compute_window_confidence(
            pred_vec=pred_vec,
            neighbor_sim=sim,
            window=win,
            full_lengths=full_lengths,
            top_signal_k=signal_top_k,
        )
        motif_group_stats = mod.summarize_motif_neighborhood(
            order=order,
            sim=sim,
            train_ids=train_ids,
            motif_groups=train_motif_groups,
            topn=motif_group_topn,
        )
        query_domains = mod.choose_query_domain(win.protein_id, domain_map=domain_map, prefer_domains=prefer_domain_tokens)
        prior_vec = None
        prior_ids: List[str] = []
        prior_corr = 0.0
        prior_cohesion = 0.0
        prior_alpha = 0.0
        prior_source = "none"
        domain_pool = int(len(order))
        fused_vec = pred_vec
        if not disable_neighbor_prior:
            prior_vec, prior_ids, prior_corr, prior_cohesion, domain_pool = mod.build_neighbor_prior(
                pred_vec=pred_vec,
                train_labels=train_labels,
                train_ids=train_ids,
                order=order,
                sim=sim,
                query_domains=query_domains,
                domain_map=domain_map,
                domain_filter_mode=domain_filter_mode,
                topn=prior_top_neighbors,
            )
            if prior_vec is not None:
                prior_alpha = mod.compute_prior_alpha(
                    confidence_components=confidence_components,
                    prior_corr=prior_corr,
                    prior_cohesion=prior_cohesion,
                    max_alpha=prior_alpha_max,
                    corr_floor=prior_corr_floor,
                    min_cohesion=prior_min_cohesion,
                )
                fused_vec = (1.0 - prior_alpha) * pred_vec + prior_alpha * prior_vec
                if prior_alpha > 0:
                    prior_source = "domain_neighbor"
            if (
                motif_group_prior_alpha_max > 0
                and str(motif_group_stats["motif_group"]) in {"UC-rich", "G-rich"}
            ):
                motif_group = str(motif_group_stats["motif_group"])
                opposite_weight = (
                    float(motif_group_stats["motif_group_g_weight"])
                    if motif_group == "UC-rich"
                    else float(motif_group_stats["motif_group_uc_weight"])
                )
                group_prior_vec, group_prior_ids, group_prior_corr, group_prior_cohesion, group_domain_pool = mod.build_motif_group_prior(
                    pred_vec=pred_vec,
                    train_labels=train_labels,
                    train_ids=train_ids,
                    order=order,
                    sim=sim,
                    motif_groups=train_motif_groups,
                    target_group=motif_group,
                    query_domains=query_domains,
                    domain_map=domain_map,
                    domain_filter_mode=domain_filter_mode,
                    topn=motif_group_prior_top_neighbors,
                    min_sim=motif_group_prior_min_sim,
                    allowed_train_ids=uc_repeat_train_ids if uc_repeat_prior_only and motif_group == "UC-rich" else None,
                )
                if group_prior_vec is not None:
                    group_alpha = mod.compute_motif_group_prior_alpha(
                        motif_group_score=float(motif_group_stats["motif_group_score"]),
                        opposite_group_weight=opposite_weight,
                        prior_cohesion=group_prior_cohesion,
                        max_alpha=motif_group_prior_alpha_max,
                        min_group_score=motif_group_prior_min_score,
                        max_opposite_weight=motif_group_prior_max_opposite,
                        min_cohesion=motif_group_prior_min_cohesion,
                    )
                    if group_alpha > prior_alpha:
                        prior_vec = group_prior_vec
                        prior_ids = group_prior_ids
                        prior_corr = group_prior_corr
                        prior_cohesion = group_prior_cohesion
                        prior_alpha = group_alpha
                        domain_pool = group_domain_pool
                        prior_source = f"motif_group:{motif_group}"
                        fused_vec = (1.0 - prior_alpha) * pred_vec + prior_alpha * prior_vec
        all_window_predictions[win.protein_id].append(
            mod.WindowPrediction(
                checkpoint="unified",
                window=win,
                pred_vec=pred_vec,
                latent_vec=latent_vec,
                neighbor_order=order,
                neighbor_sim=sim,
                confidence=confidence,
                confidence_components=confidence_components,
                fused_vec=fused_vec,
                prior_vec=prior_vec,
                prior_alpha=prior_alpha,
                prior_corr=prior_corr,
                prior_cohesion=prior_cohesion,
                prior_neighbor_ids=prior_ids,
                prior_source=prior_source,
                domain_query="|".join(sorted(query_domains)),
                domain_filter_mode=domain_filter_mode,
                domain_neighbor_pool=domain_pool,
                motif_group=str(motif_group_stats["motif_group"]),
                motif_group_score=float(motif_group_stats["motif_group_score"]),
                motif_group_margin=float(motif_group_stats["motif_group_margin"]),
                motif_group_uc_weight=float(motif_group_stats["motif_group_uc_weight"]),
                motif_group_g_weight=float(motif_group_stats["motif_group_g_weight"]),
                motif_group_other_weight=float(motif_group_stats["motif_group_other_weight"]),
            )
        )

    conflict_rows = []
    selected_rows = []
    candidate_rows = []
    selection_by_protein = {}
    prior_signature_counts: Dict[str, int] = defaultdict(int)
    for protein_id in query_ids:
        rows = list(all_window_predictions.get(protein_id, []))
        conflict_info = mod.detect_motif_group_conflict(
            rows=rows,
            min_group_score=motif_conflict_min_group_score,
            min_confidence=motif_conflict_min_confidence,
        )
        conflict_rows.append({"rbp_id": protein_id, **conflict_info})
        if conflict_info["motif_group_conflict"] and motif_conflict_mode == "penalize":
            for row in rows:
                row.confidence *= float(motif_conflict_penalty)
        if not rows:
            selection_by_protein[protein_id] = ([], conflict_info, None)
            continue
        if select_top_windows <= 0:
            raise ValueError("select_top_windows must be positive")
        if select_top_windows and conflict_info["motif_group_conflict"]:
            selected = mod.select_windows_conflict_aware(
                rows=rows,
                top_n=select_top_windows,
                min_confidence=min_confidence,
                conflict_info=conflict_info,
                domain_boundary_bonus=conflict_domain_boundary_bonus,
            )
        else:
            selected = mod.select_windows_for_fusion(
                rows=rows,
                top_n=select_top_windows,
                min_confidence=min_confidence,
            )
        final_motif_group_prior_cap = (
            None
            if conflict_info["motif_group_conflict"]
            else motif_group_prior_nonconflict_alpha_max
        )
        selection_by_protein[protein_id] = (selected, conflict_info, final_motif_group_prior_cap)
        if not conflict_info["motif_group_conflict"]:
            seen_signatures = set()
            for row in selected:
                signature = mod.prior_neighbor_signature(row, template_prior_signature_top_ids)
                if signature:
                    seen_signatures.add(signature)
            for signature in seen_signatures:
                prior_signature_counts[signature] += 1

    fused_by_protein = {}
    candidate_by_protein = {}
    for protein_id in query_ids:
        selected, conflict_info, final_motif_group_prior_cap = selection_by_protein[protein_id]
        rows = all_window_predictions.get(protein_id, [])
        if not selected:
            continue
        fused_by_protein[protein_id] = mod.fuse_predictions_with_prior_cap(
            selected,
            motif_group_prior_alpha_cap=final_motif_group_prior_cap,
            template_prior_alpha_cap=motif_group_prior_template_alpha_max,
            prior_signature_counts=prior_signature_counts,
            prior_signature_top_ids=template_prior_signature_top_ids,
            min_shared_prior_signature=template_collapse_min_shared_prior,
        )
        if export_conflict_candidates and conflict_info["motif_group_conflict"]:
            for branch_name, motif_group in [("UC_candidate", "UC-rich"), ("G_candidate", "G-rich")]:
                branch_rows = mod.select_windows_by_motif_group(
                    rows=rows,
                    motif_group=motif_group,
                    top_n=candidate_top_windows,
                    min_confidence=min_confidence,
                    domain_boundary_bonus=conflict_domain_boundary_bonus,
                )
                if not branch_rows:
                    continue
                branch_protein_id = f"{protein_id}_{branch_name}"
                candidate_by_protein[branch_protein_id] = mod.fuse_predictions(branch_rows)
                for rank0, row in enumerate(branch_rows, start=1):
                    candidate_rows.append(
                        {
                            "rbp_id": protein_id,
                            "candidate_id": branch_protein_id,
                            "candidate_branch": branch_name,
                            "candidate_motif_group": motif_group,
                            "selected_rank": rank0,
                            "window_id": row.window.window_id,
                            "window_start": row.window.start,
                            "window_end": row.window.end,
                            "window_len": len(row.window.sequence),
                            "window_source": row.window.source,
                            "window_domain": row.window.domain,
                            "confidence": row.confidence,
                            "conflict_aware_score": mod.conflict_aware_selection_score(row, conflict_domain_boundary_bonus),
                            "prior_alpha": row.prior_alpha,
                            "prior_corr": row.prior_corr,
                            "prior_cohesion": row.prior_cohesion,
                            "prior_neighbor_ids": "|".join(row.prior_neighbor_ids),
                            "prior_source": row.prior_source,
                            "motif_group": row.motif_group,
                            "motif_group_score": row.motif_group_score,
                            "motif_group_margin": row.motif_group_margin,
                            "motif_group_uc_weight": row.motif_group_uc_weight,
                            "motif_group_g_weight": row.motif_group_g_weight,
                            "motif_group_other_weight": row.motif_group_other_weight,
                            **row.confidence_components,
                        }
                    )
        for rank0, row in enumerate(selected, start=1):
            prior_signature = mod.prior_neighbor_signature(row, template_prior_signature_top_ids)
            shared_prior_signature_count = prior_signature_counts.get(prior_signature, 0) if prior_signature else 0
            effective_prior_alpha = mod.adjusted_prior_alpha(
                row=row,
                motif_group_prior_alpha_cap=final_motif_group_prior_cap,
                template_prior_alpha_cap=motif_group_prior_template_alpha_max,
                shared_prior_signature_count=shared_prior_signature_count,
                min_shared_prior_signature=template_collapse_min_shared_prior,
            )
            selected_rows.append(
                {
                    "rbp_id": protein_id,
                    "selected_rank": rank0,
                    "window_id": row.window.window_id,
                    "window_start": row.window.start,
                    "window_end": row.window.end,
                    "window_len": len(row.window.sequence),
                    "window_source": row.window.source,
                    "window_domain": row.window.domain,
                    "confidence": row.confidence,
                    "fusion_weight_basis": max(row.confidence, 1e-4),
                    "selection_mode": "conflict_aware",
                    "conflict_aware_score": mod.conflict_aware_selection_score(row, conflict_domain_boundary_bonus),
                    "prior_alpha": row.prior_alpha,
                    "final_prior_alpha": effective_prior_alpha,
                    "final_motif_group_prior_alpha_cap": final_motif_group_prior_cap,
                    "prior_corr": row.prior_corr,
                    "prior_cohesion": row.prior_cohesion,
                    "prior_neighbor_ids": "|".join(row.prior_neighbor_ids),
                    "prior_neighbor_signature": prior_signature,
                    "shared_prior_signature_count": shared_prior_signature_count,
                    "prior_source": row.prior_source,
                    "domain_query": row.domain_query,
                    "domain_filter_mode": row.domain_filter_mode,
                    "domain_neighbor_pool": row.domain_neighbor_pool,
                    "motif_group_conflict": conflict_info["motif_group_conflict"],
                    "conflict_groups": conflict_info["conflict_groups"],
                    "best_uc_window": conflict_info["best_uc_window"],
                    "best_g_window": conflict_info["best_g_window"],
                    "motif_group": row.motif_group,
                    "motif_group_score": row.motif_group_score,
                    "motif_group_margin": row.motif_group_margin,
                    "motif_group_uc_weight": row.motif_group_uc_weight,
                    "motif_group_g_weight": row.motif_group_g_weight,
                    "motif_group_other_weight": row.motif_group_other_weight,
                    **row.confidence_components,
                }
            )

    fused_by_protein = align_profile_map(fused_by_protein, kmers_train, kmers_target)
    candidate_by_protein = align_profile_map(candidate_by_protein, kmers_train, kmers_target)
    selected_df = pd.DataFrame(selected_rows)
    conflict_df = pd.DataFrame(conflict_rows)
    candidate_df = pd.DataFrame(candidate_rows)
    summary_df = pd.DataFrame(
        [
            {
                "rbp_id": protein_id,
                "query_protein_id": cfg.rbp_id_to_query_protein_id[protein_id],
                "resolved_kingdom_label": int(protein_kingdom_lookup.get(protein_id, int(query_kingdom_label))),
                "resolved_kingdom_name": KINGDOM_LABEL_TO_NAME.get(
                    int(protein_kingdom_lookup.get(protein_id, int(query_kingdom_label))),
                    "other",
                ),
                "teacher_mode": "v6_faithful_auto_teacher",
                "teacher_has_conflict": int(conflict_info["motif_group_conflict"]),
                "teacher_primary_group": str(conflict_info["conflict_groups"]).split("|")[0] if conflict_info["conflict_groups"] else "",
                "teacher_secondary_group": str(conflict_info["conflict_groups"]).split("|")[1] if "|" in str(conflict_info["conflict_groups"]) else "",
                "teacher_primary_window": conflict_info["best_uc_window"] or conflict_info["best_g_window"],
                "teacher_secondary_window": conflict_info["best_g_window"] if conflict_info["best_uc_window"] else "",
                "teacher_top_window": selected[0].window.window_id if selected else "",
                "teacher_top_window_confidence": float(selected[0].confidence) if selected else float("nan"),
                "teacher_selected_n_windows": int(len(selected)),
            }
            for protein_id, (selected, conflict_info, _) in selection_by_protein.items()
        ]
    )

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        mod.write_prediction_matrix(str(out_dir / "auto_teacher_fused_z_matrix.csv"), kmers_target, fused_by_protein)
        selected_df.to_csv(out_dir / "auto_teacher_selected_windows.csv", sep="\t", index=False)
        conflict_df.to_csv(out_dir / "auto_teacher_conflicts.tsv", sep="\t", index=False)
        summary_df.to_csv(out_dir / "auto_teacher_summary.tsv", sep="\t", index=False)
        if export_conflict_candidates and candidate_by_protein:
            mod.write_prediction_matrix(str(out_dir / "auto_teacher_conflict_candidate_z_matrix.csv"), kmers_target, candidate_by_protein)
            candidate_df.to_csv(out_dir / "auto_teacher_conflict_candidate_windows.csv", sep="\t", index=False)
    return fused_by_protein, summary_df, selected_df, conflict_df
