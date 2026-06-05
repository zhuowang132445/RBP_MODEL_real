#!/usr/bin/env python3
from __future__ import annotations

import csv
import gzip
import json
import hashlib
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path("/public/home/wz/workplace/cursor/modle/unified_rbp_model_v1")
RESULTS_DIR = PROJECT_ROOT / "results"
ALIGNMENT_TSV = RESULTS_DIR / "alignment_audit.tsv"
REPORT_JSON = RESULTS_DIR / "alignment_audit_report.json"
sys.path.insert(0, str(PROJECT_ROOT))


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return open(path, "rt", encoding="utf-8", errors="ignore")


def normalize_rna_for_fingerprint(seq: str) -> str:
    seq = str(seq).replace(" ", "").replace("\n", "").upper().replace("T", "U")
    allowed = set("ACGUN")
    return "".join(base if base in allowed else "N" for base in seq)


def sha256_update_seq(hasher, seq: str) -> None:
    hasher.update(normalize_rna_for_fingerprint(seq).encode("ascii"))
    hasher.update(b"\n")


def iter_window_rows(path: Path) -> Iterable[dict[str, str]]:
    with open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            yield row


def decode_rna_codes(arr: np.ndarray) -> list[str]:
    alphabet = np.asarray(["A", "C", "G", "U", "N"], dtype=object)
    clipped = np.clip(arr.astype(np.int64), 0, 4)
    seqs = []
    for row in clipped:
        seqs.append("".join(alphabet[row].tolist()))
    return seqs


@dataclass
class ProteinAlignmentRow:
    embedding_row: int
    binding_protein_id: str
    motif_cache_index_row: int | None
    motif_cache_protein_id: str | None
    motif_cache_embedding_row: int | None
    protein_id_match: bool
    embedding_row_match: bool
    train_usage_count: int
    val_usage_count: int
    test_usage_count: int
    used_any_split: bool
    issue: str


def audit_kingdom_labels(report: dict) -> None:
    train_text = (PROJECT_ROOT / "train_unified_original_logic.py").read_text(encoding="utf-8")
    predict_text = (PROJECT_ROOT / "predict_unified_original_logic.py").read_text(encoding="utf-8")
    auto_text = (PROJECT_ROOT / "unified_auto_teacher.py").read_text(encoding="utf-8")
    import train_unified_original_logic as train_mod
    import predict_unified_original_logic as pred_mod

    samples = {
        "Viridiplantae": int(train_mod.kingdom_label("Viridiplantae")),
        "Fungi": int(train_mod.kingdom_label("Fungi")),
        "Metazoa": int(train_mod.kingdom_label("Metazoa")),
        "Unknown": int(train_mod.kingdom_label("Unknown")),
        "ARATH_query": int(pred_mod.kingdom_label_from_query_id("ARATH|GRP7|Q03250|Glycine-rich_RNA-binding_protein_7")),
        "ORYSJ_query": int(pred_mod.kingdom_label_from_query_id("ORYSJ|LOC_Os05g24160.1|Q0DJA3|original_rice7")),
        "DROME_query": int(pred_mod.kingdom_label_from_query_id("DROME|AUB|O76922|aubergine")),
    }
    parser_default = None
    for line in predict_text.splitlines():
        if "--auto-teacher-query-kingdom-label" in line and "default=" in line:
            parser_default = line.strip()
            break
    fn_default = None
    for line in auto_text.splitlines():
        if "query_kingdom_label:" in line and " = " in line:
            fn_default = line.strip()
            break
    findings = []
    if samples["Viridiplantae"] != 1 or samples["ARATH_query"] != 1 or samples["ORYSJ_query"] != 1:
        findings.append("plant label mapping is inconsistent")
    if parser_default and "default=2" in parser_default:
        findings.append("predict parser default auto-teacher kingdom label is 2, not 1 for plants")
    if fn_default and "= 2" in fn_default:
        findings.append("unified_auto_teacher default query_kingdom_label is 2, not 1 for plants")
    report["kingdom_label_audit"] = {
        "samples": samples,
        "predict_parser_default_line": parser_default,
        "auto_teacher_default_line": fn_default,
        "findings": findings,
    }


def audit_motif_embedding_alignment(report: dict) -> None:
    import unified_config as cfg_mod

    cfg = cfg_mod.get_config()
    motif_index = pd.read_csv(PROJECT_ROOT / "data/original_logic_cache/motif_feature_index.tsv", sep="\t")
    binding_index = pd.read_csv(Path(cfg.binding_project_dir) / "08_embeddings/esm2_t33_650M_strict_fixed/protein_embedding_index.tsv", sep="\t")
    report_rows = []
    split_counts: dict[str, Counter[int]] = {}
    cache_dir = Path(json.loads((PROJECT_ROOT / "data/multitask/multitask_data_report.json").read_text(encoding="utf-8"))["sequence_cache_dir"])
    for split in ("train", "val", "test"):
        rows = np.load(cache_dir / f"{split}.protein_rows.npy", mmap_mode="r")
        cnt = Counter(np.asarray(rows, dtype=np.int64).tolist())
        split_counts[split] = cnt
    binding_index = binding_index.copy()
    if "embedding_row" not in binding_index.columns:
        binding_index["embedding_row"] = np.arange(len(binding_index), dtype=np.int64)
    binding_index["embedding_row"] = binding_index["embedding_row"].astype(int)
    motif_train = motif_index[motif_index["source"].astype(str) == "music_clip_train"].copy()
    motif_train["embedding_row"] = motif_train["embedding_row"].astype(int)
    motif_by_row = motif_train.set_index("embedding_row", drop=False).to_dict(orient="index")
    motif_by_pid = motif_train.set_index("protein_id", drop=False).to_dict(orient="index")

    issues = {
        "binding_embedding_row_unique": bool(binding_index["embedding_row"].is_unique),
        "binding_embedding_row_contiguous": set(binding_index["embedding_row"].tolist()) == set(range(len(binding_index))),
        "motif_train_embedding_row_unique": bool(motif_train["embedding_row"].is_unique),
        "motif_train_embedding_row_contiguous": set(motif_train["embedding_row"].tolist()) == set(range(len(motif_train))),
        "binding_n_rows": int(len(binding_index)),
        "motif_train_n_rows": int(len(motif_train)),
    }

    for row in binding_index.itertuples(index=False):
        emb_row = int(row.embedding_row)
        protein_id = str(row.protein_id)
        motif_row = motif_by_row.get(emb_row)
        motif_pid_row = motif_by_pid.get(protein_id)
        protein_id_match = motif_row is not None and str(motif_row["protein_id"]) == protein_id
        embedding_row_match = motif_pid_row is not None and int(motif_pid_row["embedding_row"]) == emb_row
        issue_bits = []
        if motif_row is None:
            issue_bits.append("missing_in_motif_by_row")
        if motif_pid_row is None:
            issue_bits.append("missing_in_motif_by_pid")
        if motif_row is not None and not protein_id_match:
            issue_bits.append("protein_id_mismatch")
        if motif_pid_row is not None and not embedding_row_match:
            issue_bits.append("embedding_row_mismatch")
        report_rows.append(
            ProteinAlignmentRow(
                embedding_row=emb_row,
                binding_protein_id=protein_id,
                motif_cache_index_row=int(motif_row["embedding_row"]) if motif_row is not None else None,
                motif_cache_protein_id=str(motif_row["protein_id"]) if motif_row is not None else None,
                motif_cache_embedding_row=int(motif_pid_row["embedding_row"]) if motif_pid_row is not None else None,
                protein_id_match=protein_id_match,
                embedding_row_match=embedding_row_match,
                train_usage_count=int(split_counts["train"].get(emb_row, 0)),
                val_usage_count=int(split_counts["val"].get(emb_row, 0)),
                test_usage_count=int(split_counts["test"].get(emb_row, 0)),
                used_any_split=bool(
                    split_counts["train"].get(emb_row, 0)
                    or split_counts["val"].get(emb_row, 0)
                    or split_counts["test"].get(emb_row, 0)
                ),
                issue=";".join(issue_bits),
            )
        )
    out_frame = pd.DataFrame([asdict(r) for r in report_rows]).sort_values("embedding_row")
    out_frame.to_csv(ALIGNMENT_TSV, sep="\t", index=False)

    protein_rows_findings = {}
    for split in ("train", "val", "test"):
        used_rows = sorted(split_counts[split].keys())
        min_row = int(min(used_rows)) if used_rows else None
        max_row = int(max(used_rows)) if used_rows else None
        out_of_range = sum(1 for x in used_rows if x < 0 or x >= len(binding_index))
        missing_in_motif = sum(1 for x in used_rows if x not in motif_by_row)
        pid_mismatch = sum(
            1
            for x in used_rows
            if x in motif_by_row and str(motif_by_row[x]["protein_id"]) != str(binding_index.iloc[x]["protein_id"])
        )
        protein_rows_findings[split] = {
            "n_unique_rows": int(len(used_rows)),
            "min_row": min_row,
            "max_row": max_row,
            "out_of_range_rows": int(out_of_range),
            "missing_in_motif_by_row": int(missing_in_motif),
            "protein_id_mismatch_at_same_row": int(pid_mismatch),
        }

    dataset_logic_issue = (
        not out_frame["protein_id_match"].all()
        or not out_frame["embedding_row_match"].all()
        or any(v["out_of_range_rows"] > 0 or v["missing_in_motif_by_row"] > 0 or v["protein_id_mismatch_at_same_row"] > 0 for v in protein_rows_findings.values())
    )
    report["motif_embedding_alignment_audit"] = {
        "summary": {
            **issues,
            "all_protein_id_match": bool(out_frame["protein_id_match"].all()),
            "all_embedding_row_match": bool(out_frame["embedding_row_match"].all()),
            "rows_with_any_issue": int((out_frame["issue"].astype(str) != "").sum()),
            "binding_dataset_self_motif_indexing_safe": not dataset_logic_issue,
            "binding_dataset_note": "BindingDataset uses self.motif[prow]; this is safe only if music_clip_train rows are exactly aligned to binding embedding_row order.",
        },
        "protein_rows_findings": protein_rows_findings,
        "alignment_tsv": str(ALIGNMENT_TSV),
    }


def audit_prediction_structure_assets(report: dict) -> None:
    assets = []
    pred_assets_root = PROJECT_ROOT / "data/prediction_assets"
    for asset_dir in sorted(x for x in pred_assets_root.iterdir() if x.is_dir()):
        window_candidates = sorted((asset_dir / "input").glob("*windows_w200_s50.tsv.gz"))
        meta_path = asset_dir / "structure_cache/transcriptome.structure_meta.json"
        npy_path = asset_dir / "structure_cache/transcriptome.paired_probability.npy"
        if not window_candidates or not meta_path.exists() or not npy_path.exists():
            continue
        window_path = window_candidates[0]
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        arr = np.load(npy_path, mmap_mode="r")
        key_counter = Counter()
        seqhash_counter = Counter()
        row_count = 0
        seq_hasher = hashlib.sha256()
        min_start = None
        max_start = None
        len_ok = True
        for row in iter_window_rows(window_path):
            row_count += 1
            ws = int(row["window_start"])
            we = int(row["window_end"])
            seq = str(row["rna_seq"])
            seq_hash = hashlib.sha1(seq.encode("utf-8")).hexdigest()
            key = (str(row["transcript_id"]), ws, we)
            key_counter[key] += 1
            seqhash_counter[(str(row["transcript_id"]), ws, we, seq_hash)] += 1
            sha256_update_seq(seq_hasher, seq)
            if len(seq) != int(meta["rna_len"]):
                len_ok = False
            min_start = ws if min_start is None else min(min_start, ws)
            max_start = ws if max_start is None else max(max_start, ws)
        duplicate_keys = sum(v - 1 for v in key_counter.values() if v > 1)
        duplicate_key_seq = sum(v - 1 for v in seqhash_counter.values() if v > 1)
        row_hash = seq_hasher.hexdigest()
        assets.append(
            {
                "asset": asset_dir.name,
                "window_tsv": str(window_path),
                "structure_npy": str(npy_path),
                "structure_meta": str(meta_path),
                "window_rows": int(row_count),
                "structure_rows": int(arr.shape[0]),
                "rna_len_window_meta": int(meta["rna_len"]),
                "structure_width": int(arr.shape[1]),
                "row_count_match": bool(int(row_count) == int(arr.shape[0])),
                "row_order_hash_match": bool(row_hash == str(meta.get("rna_row_order_sha256"))),
                "duplicate_key_count": int(duplicate_keys),
                "duplicate_key_with_same_seqhash_count": int(duplicate_key_seq),
                "matched_rows_inferred": int(row_count if row_hash == str(meta.get("rna_row_order_sha256")) and row_count == int(arr.shape[0]) else 0),
                "unmatched_rows_inferred": int(0 if row_hash == str(meta.get("rna_row_order_sha256")) and row_count == int(arr.shape[0]) else abs(int(row_count) - int(arr.shape[0]))),
                "window_start_min": int(min_start) if min_start is not None else None,
                "window_start_max": int(max_start) if max_start is not None else None,
                "window_coordinates_look_1_based": bool(min_start == 1),
                "window_lengths_match_meta": bool(len_ok),
            }
        )
    report["prediction_structure_cache_audit"] = assets


def audit_coordinate_offset(report: dict) -> None:
    rice_window_tsv = PROJECT_ROOT / "data/prediction_assets/rice_v7/input/rice_v7_windows_w200_s50.tsv.gz"
    observed_tsv = Path("/public/home/wz/workplace/cursor/RBP_clip_data/MuSIC_data/11_rice_prediction/validation_hice_window_osdrb1_matched/hice_centered_observed_scores.tsv")
    bg_tsv = Path("/public/home/wz/workplace/cursor/RBP_clip_data/MuSIC_data/11_rice_prediction/validation_hice_window_osdrb1_matched/hice_same_transcript_background_scores.tsv.gz")
    if not rice_window_tsv.exists() or not observed_tsv.exists() or not bg_tsv.exists():
        return
    direct_obs = set()
    plus1_obs = set()
    with open(observed_tsv, newline="", encoding="utf-8", errors="ignore") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if row["model_name"] != "full_length":
                continue
            tid = str(row["transcript_id"])
            if not tid:
                continue
            s0 = int(row["observed_window_start"])
            e = int(row["observed_window_end"])
            direct_obs.add((tid, s0, e))
            plus1_obs.add((tid, s0 + 1, e))
    direct_bg = set()
    plus1_bg = set()
    with gzip.open(bg_tsv, "rt", encoding="utf-8", errors="ignore") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if row["model_name"] != "full_length":
                continue
            tid = str(row["transcript_id"])
            if not tid:
                continue
            s0 = int(row["background_window_start"])
            e = int(row["background_window_end"])
            direct_bg.add((tid, s0, e))
            plus1_bg.add((tid, s0 + 1, e))
    counts = {
        "observed_direct_match": 0,
        "observed_plus1_match": 0,
        "background_direct_match": 0,
        "background_plus1_match": 0,
    }
    with gzip.open(rice_window_tsv, "rt", encoding="utf-8", errors="ignore") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            key = (str(row["transcript_id"]), int(row["window_start"]), int(row["window_end"]))
            if key in direct_obs:
                counts["observed_direct_match"] += 1
            if key in plus1_obs:
                counts["observed_plus1_match"] += 1
            if key in direct_bg:
                counts["background_direct_match"] += 1
            if key in plus1_bg:
                counts["background_plus1_match"] += 1
    report["coordinate_offset_audit"] = counts


def audit_kmers(report: dict) -> None:
    kmers = np.load(PROJECT_ROOT / "data/multitask/motif_profiles.npz")["kmers"].astype(str)
    normalized = np.char.upper(np.char.replace(kmers.astype(str), "T", "U"))
    allowed = set("ACGU")
    bad_chars = sorted({ch for k in normalized.tolist() for ch in k if ch not in allowed})
    report["kmer_normalization_audit"] = {
        "n_kmers": int(len(kmers)),
        "contains_lowercase_before_normalization": bool(any(k != k.upper() for k in kmers.tolist())),
        "contains_T_before_normalization": bool(any("T" in k for k in kmers.tolist())),
        "all_uppercase_rna_after_normalization": bool(np.all(normalized == np.char.upper(normalized))),
        "all_acgu_after_normalization": bool(len(bad_chars) == 0),
        "bad_chars_after_normalization": bad_chars,
        "example_before": kmers[:5].tolist(),
        "example_after": normalized[:5].tolist(),
    }


def audit_gene_ranking_bias(report: dict) -> None:
    items = []
    for gene_path in sorted((PROJECT_ROOT / "results").glob("*/gene_scores.tsv")):
        try:
            frame = pd.read_csv(gene_path, sep="\t")
        except Exception as exc:
            items.append({"gene_scores_tsv": str(gene_path), "error": str(exc)})
            continue
        required = {"n_windows", "max_score", "mean_top3_score", "mean_top5_score"}
        if not required.issubset(set(frame.columns)):
            items.append({"gene_scores_tsv": str(gene_path), "missing_columns": sorted(required - set(frame.columns))})
            continue
        subrows = []
        for rbp_id, sub in frame.groupby("rbp_id", sort=False):
            corr = {
                "gene_scores_tsv": str(gene_path),
                "result_dir": gene_path.parent.name,
                "rbp_id": str(rbp_id),
                "n_rows": int(len(sub)),
                "pearson_nwindows_max_score": float(sub["n_windows"].corr(sub["max_score"], method="pearson")),
                "pearson_nwindows_top3": float(sub["n_windows"].corr(sub["mean_top3_score"], method="pearson")),
                "pearson_nwindows_top5": float(sub["n_windows"].corr(sub["mean_top5_score"], method="pearson")),
                "spearman_nwindows_max_score": float(sub["n_windows"].corr(sub["max_score"], method="spearman")),
                "spearman_nwindows_top3": float(sub["n_windows"].corr(sub["mean_top3_score"], method="spearman")),
                "spearman_nwindows_top5": float(sub["n_windows"].corr(sub["mean_top5_score"], method="spearman")),
            }
            corr["max_score_bias_flag"] = bool(
                abs(corr["pearson_nwindows_max_score"]) >= 0.2
                or abs(corr["spearman_nwindows_max_score"]) >= 0.2
            )
            subrows.append(corr)
        items.extend(subrows)
    report["gene_ranking_bias_audit"] = items


def audit_trainability(report: dict) -> None:
    import unified_config as cfg_mod
    import train_unified_original_logic as train_mod
    import unified_original_logic_model as model_mod

    cfg = cfg_mod.get_config()
    model = model_mod.OriginalLogicUnifiedRBPModel(cfg=cfg, load_pretrained=False).cpu()
    stages = {}
    for stage in ("binding", "late_fusion", "joint"):
        train_mod.set_stage_trainability(model, stage)
        rows = []
        total = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                total += int(param.numel())
                prefix = name.split(".", 1)[0]
                rows.append((name, int(param.numel()), prefix))
        by_prefix = Counter(prefix for _, _, prefix in rows)
        by_prefix_params = {}
        for name, numel, prefix in rows:
            by_prefix_params[prefix] = by_prefix_params.get(prefix, 0) + numel
        stages[stage] = {
            "trainable_parameter_count": int(total),
            "trainable_prefix_counts": {k: int(v) for k, v in sorted(by_prefix.items())},
            "trainable_prefix_numel": {k: int(v) for k, v in sorted(by_prefix_params.items())},
            "binding_model_any_trainable": bool(any(n.startswith("binding_model.") for n, _, _ in rows)),
            "binding_model_protein_embedding_trainable": bool(any(n.startswith("binding_model.protein_embedding.") for n, _, _ in rows)),
            "motif_esm_any_trainable": bool(any(n.startswith("motif_model.esm2.") for n, _, _ in rows)),
        }
    report["trainability_audit"] = stages


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "project_root": str(PROJECT_ROOT),
    }
    audit_kingdom_labels(report)
    audit_motif_embedding_alignment(report)
    audit_prediction_structure_assets(report)
    audit_coordinate_offset(report)
    audit_kmers(report)
    audit_gene_ranking_bias(report)
    audit_trainability(report)
    REPORT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(str(ALIGNMENT_TSV))
    print(str(REPORT_JSON))


if __name__ == "__main__":
    main()
