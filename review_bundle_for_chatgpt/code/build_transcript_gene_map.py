#!/usr/bin/env python3
"""Build transcript_id -> gene_id map from GTF or GFF3 annotation."""

from __future__ import annotations

import argparse
import gzip
from pathlib import Path


def open_text(path: Path):
    return gzip.open(path, "rt", encoding="utf-8", errors="ignore") if path.suffix == ".gz" else open(path, "rt", encoding="utf-8", errors="ignore")


def parse_gtf_attrs(raw: str) -> dict[str, str]:
    attrs = {}
    for item in raw.strip().split(";"):
        item = item.strip()
        if not item or " " not in item:
            continue
        key, value = item.split(" ", 1)
        attrs[key.strip()] = value.strip().strip('"')
    return attrs


def parse_gff3_attrs(raw: str) -> dict[str, str]:
    attrs = {}
    for item in raw.strip().split(";"):
        item = item.strip()
        if not item or "=" not in item:
            continue
        key, value = item.split("=", 1)
        attrs[key.strip()] = value.strip()
    return attrs


def build_map(annotation_path: Path) -> list[tuple[str, str]]:
    suffixes = annotation_path.name.lower()
    is_gtf = suffixes.endswith(".gtf") or suffixes.endswith(".gtf.gz")
    pairs: dict[str, str] = {}
    with open_text(annotation_path) as handle:
        for raw_line in handle:
            if not raw_line or raw_line.startswith("#"):
                continue
            line = raw_line.rstrip("\n")
            parts = line.split("\t")
            if len(parts) < 9:
                continue
            feature = parts[2]
            attrs = parse_gtf_attrs(parts[8]) if is_gtf else parse_gff3_attrs(parts[8])
            transcript_id = None
            gene_id = None
            if is_gtf:
                if feature != "transcript":
                    continue
                transcript_id = attrs.get("transcript_id")
                gene_id = attrs.get("gene_id")
            else:
                if feature not in {"mRNA", "transcript", "lnc_RNA", "ncRNA", "tRNA", "rRNA", "miRNA", "snoRNA", "snRNA"}:
                    continue
                transcript_id = attrs.get("transcript_id") or attrs.get("ID")
                gene_id = attrs.get("gene_id") or attrs.get("Parent")
                if transcript_id and transcript_id.startswith("transcript:"):
                    transcript_id = transcript_id.split(":", 1)[1]
                if gene_id and gene_id.startswith("gene:"):
                    gene_id = gene_id.split(":", 1)[1]
            if transcript_id and gene_id:
                pairs[transcript_id] = gene_id
    if not pairs:
        raise ValueError(f"no transcript->gene pairs parsed from: {annotation_path}")
    return sorted(pairs.items())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", required=True, help="GTF or GFF3 annotation file, optionally gzipped.")
    parser.add_argument("--out-tsv", required=True)
    args = parser.parse_args()

    annotation_path = Path(args.annotation)
    out_path = Path(args.out_tsv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pairs = build_map(annotation_path)
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write("transcript_id\tgene_id\n")
        for transcript_id, gene_id in pairs:
            handle.write(f"{transcript_id}\t{gene_id}\n")
    print(f"built transcript->gene map: n={len(pairs)} out={out_path}", flush=True)


if __name__ == "__main__":
    main()
