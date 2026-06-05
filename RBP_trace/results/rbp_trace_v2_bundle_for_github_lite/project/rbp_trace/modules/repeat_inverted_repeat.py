from __future__ import annotations

from collections import Counter
from typing import Dict

import numpy as np


def normalize_rna(seq: str) -> str:
    return str(seq).upper().replace("T", "U")


def reverse_complement(seq: str) -> str:
    table = str.maketrans({"A": "U", "U": "A", "C": "G", "G": "C"})
    return normalize_rna(seq).translate(table)[::-1]


def kmer_reverse_complement_density(seq: str, k: int) -> float:
    seq = normalize_rna(seq)
    total = max(len(seq) - k + 1, 0)
    if total == 0:
        return float("nan")
    kmers = [seq[i : i + k] for i in range(total)]
    kset = set(kmers)
    hits = sum(1 for kmer in kmers if reverse_complement(kmer) in kset)
    return hits / total


def longest_internal_rc_match(seq: str, min_k: int = 4, max_k: int = 12) -> int:
    seq = normalize_rna(seq)
    max_k = min(max_k, len(seq))
    for k in range(max_k, min_k - 1, -1):
        total = len(seq) - k + 1
        kmers = [seq[i : i + k] for i in range(total)]
        kset = set(kmers)
        if any(reverse_complement(kmer) in kset for kmer in kmers):
            return k
    return 0


def internal_rc_pair_count(seq: str, k: int = 5) -> int:
    seq = normalize_rna(seq)
    total = max(len(seq) - k + 1, 0)
    if total == 0:
        return 0
    counts = Counter(seq[i : i + k] for i in range(total))
    seen = set()
    pair_count = 0
    for kmer, count in counts.items():
        rc = reverse_complement(kmer)
        if (kmer, rc) in seen or (rc, kmer) in seen:
            continue
        if rc in counts:
            if rc == kmer:
                pair_count += count * max(count - 1, 0) // 2
            else:
                pair_count += count * counts[rc]
        seen.add((kmer, rc))
    return int(pair_count)


def palindrome_score(seq: str) -> float:
    seq = normalize_rna(seq)
    scores = []
    for k in (4, 6):
        total = max(len(seq) - k + 1, 0)
        if total == 0:
            continue
        pal = sum(1 for i in range(total) if seq[i : i + k] == reverse_complement(seq[i : i + k]))
        scores.append(pal / total)
    return float(np.mean(scores)) if scores else float("nan")


def local_self_complementarity_score(seq: str, window: int = 40) -> float:
    seq = normalize_rna(seq)
    if len(seq) <= window:
        return float(np.nanmean([kmer_reverse_complement_density(seq, 4), kmer_reverse_complement_density(seq, 5)]))
    vals = []
    for start in range(0, len(seq) - window + 1, max(window // 2, 1)):
        sub = seq[start : start + window]
        vals.append(np.nanmean([kmer_reverse_complement_density(sub, 4), kmer_reverse_complement_density(sub, 5)]))
    return float(np.nanmax(vals)) if vals else float("nan")


def simple_repeat_score(seq: str) -> float:
    seq = normalize_rna(seq)
    length = len(seq)
    if length == 0:
        return float("nan")
    probs = np.array([seq.count(base) / length for base in "ACGU"], dtype=float)
    probs = probs[probs > 0]
    entropy = float(-(probs * np.log2(probs)).sum()) if len(probs) else 0.0
    low_complexity = 1.0 - min(entropy / 2.0, 1.0)
    homopolymer = max((len(run) for run in "".join([c if c == "U" else " " for c in seq]).split()), default=0) / length
    return float((low_complexity + homopolymer) / 2.0)


def compute_repeat_features(seq: str) -> Dict[str, float]:
    rc4 = kmer_reverse_complement_density(seq, 4)
    rc5 = kmer_reverse_complement_density(seq, 5)
    rc6 = kmer_reverse_complement_density(seq, 6)
    longest = longest_internal_rc_match(seq)
    pair_count = internal_rc_pair_count(seq, 5)
    pal = palindrome_score(seq)
    local_sc = local_self_complementarity_score(seq)
    simple = simple_repeat_score(seq)
    inv = float(np.nanmean([rc4, rc5, rc6, local_sc, min(longest / 12.0, 1.0)]))
    return {
        "reverse_complement_4mer_density": rc4,
        "reverse_complement_5mer_density": rc5,
        "reverse_complement_6mer_density": rc6,
        "longest_internal_rc_match": float(longest),
        "internal_rc_pair_count": float(pair_count),
        "palindrome_score": pal,
        "inverted_repeat_score": inv,
        "local_self_complementarity_score": local_sc,
        "simple_repeat_score": simple,
        "overlap_TE": np.nan,
        "TE_class": "NA",
        "TE_family": "NA",
        "repeat_overlap_fraction": np.nan,
        "distance_to_nearest_TE": np.nan,
    }
