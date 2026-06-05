#!/usr/bin/env python3
"""Rank a small known-kmer probe set against predicted motif outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_KNOWN = {
    "AtGRP7": [
        ("U-rich_probe", "UUUUUUU"),
        ("U-rich_probe", "UUUGAUU"),
        ("G-rich_probe", "UAGGUAG"),
        ("G-rich_probe", "AGUAGGA"),
    ],
    "AtGRP8": [
        ("U-rich_probe", "UUUUUUU"),
        ("U-rich_probe", "UUUGAUU"),
        ("G-rich_probe", "UAGGUAG"),
        ("G-rich_probe", "AGUAGGA"),
    ],
    "LOC_Os05g24160.1": [
        ("UC-rich_probe", "UUUCUCU"),
        ("UC-rich_probe", "UCUCUCU"),
        ("UC-rich_probe", "CUCUUCU"),
        ("G-rich_probe", "UAGGUAG"),
        ("G-rich_probe", "AGUAGGA"),
    ],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motif-top-kmers", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.motif_top_kmers, sep="\t")
    frame["kmer_norm"] = frame["kmer"].astype(str).str.upper().str.replace("T", "U", regex=False)
    rows = []
    for rbp_id, probes in DEFAULT_KNOWN.items():
        sub = frame[frame["rbp_id"].astype(str) == rbp_id].copy()
        if sub.empty:
            continue
        for motif_label, probe in probes:
            hit = sub[sub["kmer_norm"] == probe]
            if hit.empty:
                rows.append(
                    {
                        "rbp_id": rbp_id,
                        "motif_label": motif_label,
                        "probe_kmer": probe,
                        "direction": "",
                        "rank": pd.NA,
                        "predicted_zscore": pd.NA,
                    }
                )
            else:
                best = hit.sort_values(["direction", "rank"]).iloc[0]
                rows.append(
                    {
                        "rbp_id": rbp_id,
                        "motif_label": motif_label,
                        "probe_kmer": probe,
                        "direction": best["direction"],
                        "rank": int(best["rank"]),
                        "predicted_zscore": float(best["predicted_zscore"]),
                    }
                )
    pd.DataFrame(rows).to_csv(args.out, sep="\t", index=False)


if __name__ == "__main__":
    main()
