#!/usr/bin/env python3
"""Build a transcriptome RNA-window table for unified prediction."""

from __future__ import annotations

import argparse
import gzip
from pathlib import Path


REQUIRED_COLUMNS = ["gene_id", "transcript_id", "window_start", "window_end", "rna_seq"]


def read_fasta(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    header = None
    chunks = []
    with opener(path, "rt", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks)
                header = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)
    if header is not None:
        yield header, "".join(chunks)


def load_gene_map(path: Path) -> dict[str, str]:
    mapping = {}
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="ignore") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        if len(header) < 2:
            raise ValueError("transcript-gene map must have at least two tab-separated columns")
        key_idx = 0
        val_idx = 1
        lowered = {name.lower(): idx for idx, name in enumerate(header)}
        if "transcript_id" in lowered:
            key_idx = lowered["transcript_id"]
        if "gene_id" in lowered:
            val_idx = lowered["gene_id"]
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if max(key_idx, val_idx) >= len(parts):
                continue
            transcript_id = parts[key_idx].strip()
            gene_id = parts[val_idx].strip()
            if transcript_id and gene_id:
                mapping[transcript_id] = gene_id
    if not mapping:
        raise ValueError(f"no transcript->gene pairs loaded from: {path}")
    return mapping


def iter_windows(seq: str, window_size: int, stride: int):
    seq = str(seq).upper().replace("T", "U")
    if len(seq) < window_size:
        return
    starts = list(range(0, len(seq) - window_size + 1, stride))
    if starts[-1] != len(seq) - window_size:
        starts.append(len(seq) - window_size)
    for start0 in starts:
        end0 = start0 + window_size
        yield start0 + 1, end0, seq[start0:end0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transcript-fasta", required=True)
    parser.add_argument("--transcript-gene-map", required=True)
    parser.add_argument("--out-tsv", required=True)
    parser.add_argument("--window-size", type=int, default=200)
    parser.add_argument("--stride", type=int, default=50)
    parser.add_argument("--max-transcripts", type=int, default=None)
    args = parser.parse_args()

    if args.window_size < 1 or args.stride < 1:
        raise ValueError("--window-size and --stride must be positive")

    fasta_path = Path(args.transcript_fasta)
    gene_map_path = Path(args.transcript_gene_map)
    out_path = Path(args.out_tsv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gene_map = load_gene_map(gene_map_path)

    opener = gzip.open if out_path.suffix == ".gz" else open
    n_transcripts = 0
    n_windows = 0
    with opener(out_path, "wt", encoding="utf-8") as out_handle:
        out_handle.write("\t".join(REQUIRED_COLUMNS) + "\n")
        for transcript_id, seq in read_fasta(fasta_path):
            if args.max_transcripts is not None and n_transcripts >= args.max_transcripts:
                break
            if transcript_id not in gene_map:
                continue
            n_transcripts += 1
            gene_id = gene_map[transcript_id]
            for start, end, rna_seq in iter_windows(seq, args.window_size, args.stride):
                out_handle.write(f"{gene_id}\t{transcript_id}\t{start}\t{end}\t{rna_seq}\n")
                n_windows += 1
    print(
        f"built transcriptome window table: transcripts={n_transcripts} windows={n_windows} out={out_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
