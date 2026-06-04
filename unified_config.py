#!/usr/bin/env python3
"""Configuration for unified RBP motif + RNA-window binding model v1."""

from dataclasses import asdict, dataclass, field
from typing import Dict, List


@dataclass
class UnifiedConfig:
    work_dir: str = "/public/home/wz/workplace/cursor/modle/unified_rbp_model_v1"

    motif_snapshot_dir: str = "/public/home/wz/workplace/cursor/modle/snapshots/rbp_motif_model_v6_no_test_time_prior_prototype_20260527"
    motif_model_py: str = "/public/home/wz/workplace/cursor/modle/snapshots/rbp_motif_model_v6_no_test_time_prior_prototype_20260527/scripts/model.py"
    motif_checkpoint: str = "/public/home/wz/workplace/cursor/modle/snapshots/rbp_motif_model_v6_no_test_time_prior_prototype_20260527/checkpoints/v6_1_seed_rank_conditioned_model.pth"
    motif_model_name: str = "facebook/esm2_t30_150M_UR50D"

    binding_project_dir: str = "/public/home/wz/workplace/cursor/RBP_clip_data/MuSIC_data"
    binding_train_py: str = "/public/home/wz/workplace/cursor/RBP_clip_data/MuSIC_data/scripts/train_rbp_binding_cnn.py"
    binding_checkpoint: str = "/public/home/wz/workplace/cursor/RBP_clip_data/MuSIC_data/09_models/rbp_binding_cnn_esm2_t33_650M_strict_fixed/best_model.pt"
    binding_train_embedding_npy: str = "/public/home/wz/workplace/cursor/RBP_clip_data/MuSIC_data/08_embeddings/esm2_t33_650M_strict_fixed/protein_embeddings.npy"
    binding_train_embedding_index: str = "/public/home/wz/workplace/cursor/RBP_clip_data/MuSIC_data/08_embeddings/esm2_t33_650M_strict_fixed/protein_embedding_index.tsv"

    stage1_window_features: str = "/public/home/wz/workplace/cursor/modle/fusion_v2/cache/fusion_v2_window_features.tsv.gz"
    stage1_rbp_features_json: str = "/public/home/wz/workplace/cursor/modle/fusion_v2/cache/rbp_features.json"

    output_dir: str = "/public/home/wz/workplace/cursor/modle/unified_rbp_model_v1/results"
    checkpoint_dir: str = "/public/home/wz/workplace/cursor/modle/unified_rbp_model_v1/checkpoints"
    data_dir: str = "/public/home/wz/workplace/cursor/modle/unified_rbp_model_v1/data"
    data_sources_dir: str = "/public/home/wz/workplace/cursor/modle/unified_rbp_model_v1/data_sources"

    rna_len: int = 200
    motif_latent_dim: int = 64
    motif_subtype_dim: int = 9
    motif_seed_dim: int = 8
    binding_protein_dim: int = 1280
    fusion_hidden: int = 128
    dropout: float = 0.2

    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 50
    patience: int = 10
    seed: int = 42
    device: str = "cuda"
    group_split_column: str = "gene_id"

    rbp_id_to_query_protein_id: Dict[str, str] = field(default_factory=lambda: {
        "AtGRP7": "ARATH|GRP7|Q03250|Glycine-rich_RNA-binding_protein_7",
        "AtGRP8": "ARATH|GRP8|Q03251|Glycine-rich_RNA-binding_protein_8",
        "LOC_Os05g24160.1": "ORYSJ|LOC_Os05g24160.1|Q0DJA3|original_rice7",
        "w1": "w1",
        "w2": "w2",
        "w3": "w3",
        "w4": "w4",
        "w5": "w5",
        "w6": "w6",
    })

    required_stage1_feature_columns: List[str] = field(default_factory=lambda: [
        "motif_match_score",
        "top_kmer_count",
        "motif_density",
        "matched_unique_fraction",
        "max_matched_kmer_zscore",
        "mean_matched_kmer_zscore",
        "paired_probability_mean",
        "paired_probability_median",
        "paired_probability_max",
        "fraction_high_paired",
        "fraction_low_paired",
        "A_content",
        "C_content",
        "G_content",
        "U_content",
        "GC_content",
        "AU_content",
        "low_complexity",
        "sequence_entropy",
        "base_binding_logit",
        "base_binding_score",
    ])

    def to_dict(self) -> Dict:
        return asdict(self)


def get_config() -> UnifiedConfig:
    return UnifiedConfig()
