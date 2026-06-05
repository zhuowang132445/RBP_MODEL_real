from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd


def parse_gtf_attributes(field: str) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for part in field.strip().split(";"):
        part = part.strip()
        if not part:
            continue
        if " " in part:
            key, value = part.split(" ", 1)
            attrs[key] = value.strip().strip('"')
    return attrs


def _translate_overlap(exon_start: int, exon_end: int, tx_exon_start: int, ov_start: int, ov_end: int, strand: str) -> tuple[int, int]:
    if strand == "-":
        start = tx_exon_start + (exon_end - ov_end)
        end = tx_exon_start + (exon_end - ov_start)
    else:
        start = tx_exon_start + (ov_start - exon_start)
        end = tx_exon_start + (ov_end - exon_start)
    return int(start), int(end)


def load_region_model(gtf_path: str | Path) -> dict[str, dict]:
    transcript_data: dict[str, dict] = defaultdict(lambda: {"exon": [], "CDS": [], "5UTR": [], "3UTR": [], "biotype": None, "strand": "+"})
    with open(gtf_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9:
                continue
            feature = fields[2]
            start = int(fields[3])
            end = int(fields[4])
            strand = fields[6]
            attrs = parse_gtf_attributes(fields[8])
            transcript_id = attrs.get("transcript_id")
            if not transcript_id:
                continue
            item = transcript_data[transcript_id]
            item["strand"] = strand
            biotype = attrs.get("transcript_biotype") or attrs.get("gene_biotype") or attrs.get("transcript_type") or attrs.get("gene_type")
            if biotype and item["biotype"] is None:
                item["biotype"] = biotype
            if feature.lower() == "exon":
                item["exon"].append((start, end))
            elif feature.lower() == "cds":
                item["CDS"].append((start, end))
            elif feature.lower() in {"five_prime_utr", "5utr", "utr5"}:
                item["5UTR"].append((start, end))
            elif feature.lower() in {"three_prime_utr", "3utr", "utr3"}:
                item["3UTR"].append((start, end))

    model: dict[str, dict] = {}
    for transcript_id, item in transcript_data.items():
        exons = sorted(item["exon"], key=lambda x: x[0], reverse=item["strand"] == "-")
        tx_segments = {"CDS": [], "5UTR": [], "3UTR": []}
        tx_pos = 1
        for exon_start, exon_end in exons:
            exon_len = exon_end - exon_start + 1
            for label in ("CDS", "5UTR", "3UTR"):
                for feat_start, feat_end in item[label]:
                    ov_start = max(exon_start, feat_start)
                    ov_end = min(exon_end, feat_end)
                    if ov_start <= ov_end:
                        tx_segments[label].append(_translate_overlap(exon_start, exon_end, tx_pos, ov_start, ov_end, item["strand"]))
            tx_pos += exon_len
        model[transcript_id] = {
            "segments": tx_segments,
            "biotype": item["biotype"],
            "has_cds": bool(item["CDS"]),
        }
    return model


def annotate_window_region(transcript_id: str, start: int, end: int, region_model: dict[str, dict]) -> dict[str, object]:
    info = region_model.get(str(transcript_id))
    if info is None:
        return {"region_type": "unknown"}
    overlaps = {"CDS": 0, "5UTR": 0, "3UTR": 0}
    for label, segments in info["segments"].items():
        for seg_start, seg_end in segments:
            ov_start = max(start, seg_start)
            ov_end = min(end, seg_end)
            if ov_start <= ov_end:
                overlaps[label] += ov_end - ov_start + 1
    if sum(overlaps.values()) == 0:
        biotype = (info.get("biotype") or "").lower()
        if (not info["has_cds"]) or ("rna" in biotype and "mrna" not in biotype):
            region_type = "ncRNA"
        else:
            region_type = "unknown"
    else:
        region_type = max(overlaps, key=overlaps.get)
    return {"region_type": region_type}


def annotate_region_features(
    windows: pd.DataFrame,
    gtf_path: str | Path | None,
    out_path: str | Path,
    missing_report_path: str | Path,
) -> pd.DataFrame:
    frame = windows.copy()
    if not gtf_path or not Path(gtf_path).exists():
        frame["region_type"] = "NA"
        Path(missing_report_path).write_text("annotation_gtf missing; region features set to NA\n", encoding="utf-8")
    else:
        model = load_region_model(gtf_path)
        annotated = [
            annotate_window_region(row.transcript_id, int(row.window_start), int(row.window_end), model)
            for row in frame.itertuples(index=False)
        ]
        frame["region_type"] = [item["region_type"] for item in annotated]
    frame["is_CDS"] = (frame["region_type"] == "CDS").astype(float)
    frame["is_5UTR"] = (frame["region_type"] == "5UTR").astype(float)
    frame["is_3UTR"] = (frame["region_type"] == "3UTR").astype(float)
    frame["is_ncRNA"] = (frame["region_type"] == "ncRNA").astype(float)
    frame["is_unknown"] = (frame["region_type"].isin(["unknown", "NA"])).astype(float)
    frame.to_csv(out_path, sep="\t", index=False)
    return frame


def summarize_region_top3(windows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for gene_id, sub in windows.groupby("gene_id", sort=False):
        rows.append(
            {
                "gene_id": str(gene_id),
                "best_window_region_type": str(sub.iloc[0]["region_type"]),
                "best_window_is_CDS": float(sub.iloc[0]["is_CDS"]),
                "best_window_is_5UTR": float(sub.iloc[0]["is_5UTR"]),
                "best_window_is_3UTR": float(sub.iloc[0]["is_3UTR"]),
                "best_window_is_ncRNA": float(sub.iloc[0]["is_ncRNA"]),
                "best_window_is_unknown": float(sub.iloc[0]["is_unknown"]),
                "top3_CDS_fraction": float(sub["is_CDS"].mean()),
                "top3_UTR_fraction": float((sub["is_5UTR"] + sub["is_3UTR"]).mean()),
                "top3_ncRNA_fraction": float(sub["is_ncRNA"].mean()),
                "top3_unknown_fraction": float(sub["is_unknown"].mean()),
            }
        )
    return pd.DataFrame(rows)
