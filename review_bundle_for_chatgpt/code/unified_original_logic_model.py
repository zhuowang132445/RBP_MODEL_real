#!/usr/bin/env python3
"""Unified model that keeps the original V6.1 motif logic and MuSIC RBPBindingCNN logic."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn as nn

from unified_config import get_config


def load_module_from_path(module_name: str, path: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module




def remap_v61_state_dict_for_current_model(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Map older V6.1 Sequential indices to the current model.py layout."""
    mapped = {}
    key_map = {
        "motif_subtype_head.3.": "motif_subtype_head.4.",
        "seed_group_head.3.": "seed_group_head.4.",
        "decoder.1.": "decoder.2.",
        "decoder.3.": "decoder.5.",
    }
    for key, value in state_dict.items():
        new_key = key
        for old, new in key_map.items():
            if key.startswith(old):
                new_key = new + key[len(old):]
                break
        mapped[new_key] = value
    return mapped


def infer_motif_checkpoint_config(state_dict: Dict[str, torch.Tensor], default_model_name: str) -> Dict:
    svd_v = state_dict["svd_v"]
    subtype_dim = int(state_dict.get("subtype_condition_embedding", torch.empty(9, 32)).shape[0])
    seed_dim = int(state_dict.get("seed_condition_embedding", torch.empty(8, 32)).shape[0])
    subtype_condition_dim = int(state_dict.get("subtype_condition_embedding", torch.empty(9, 32)).shape[1])
    seed_condition_dim = int(state_dict.get("seed_condition_embedding", torch.empty(8, 32)).shape[1])
    return {
        "model_name": default_model_name,
        "output_dim": int(svd_v.shape[1]),
        "svd_dim": int(svd_v.shape[0]),
        "direction_mode": "sample",
        "num_motif_subtypes": subtype_dim,
        "num_seed_groups": seed_dim,
        "use_subtype_conditioned_decoder": "subtype_condition_embedding" in state_dict,
        "subtype_condition_dim": subtype_condition_dim,
        "seed_condition_dim": seed_condition_dim,
        "dropout": 0.10,
    }


class OriginalLogicUnifiedRBPModel(nn.Module):
    """One checkpoint containing both original models plus a motif-aware binding bridge.

    - motif_model is the original ESMRNA_Predictor implementation.
    - binding_model is the original MuSIC RBPBindingCNN implementation.
    - motif-aware binding reuses binding_model.encode_rna and binding_model.protein_projector.
    """

    def __init__(self, cfg=None, load_pretrained: bool = True, freeze_motif_esm: bool = True):
        super().__init__()
        self.cfg = cfg or get_config()
        motif_state = torch.load(self.cfg.motif_checkpoint, map_location="cpu", weights_only=False)
        motif_module = load_module_from_path("unified_original_motif_model", self.cfg.motif_model_py)
        self.motif_model = motif_module.ESMRNA_Predictor(**infer_motif_checkpoint_config(motif_state, self.cfg.motif_model_name))
        if load_pretrained:
            self.motif_model.load_state_dict(remap_v61_state_dict_for_current_model(motif_state), strict=True)
        if freeze_motif_esm:
            for p in self.motif_model.esm2.parameters():
                p.requires_grad = False
            self.motif_model.esm2.eval()

        binding_module = load_module_from_path("unified_original_binding_model", self.cfg.binding_train_py)
        train_embeddings = np.load(self.cfg.binding_train_embedding_npy).astype(np.float32)
        self.binding_model = binding_module.RBPBindingCNN(train_embeddings)
        if load_pretrained:
            ckpt = torch.load(self.cfg.binding_checkpoint, map_location="cpu", weights_only=False)
            state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
            self.binding_model.load_state_dict(state, strict=True)

        motif_dim = self.cfg.motif_latent_dim + self.cfg.motif_subtype_dim + self.cfg.motif_seed_dim
        self.motif_binding_projector = nn.Sequential(
            nn.Linear(motif_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(self.cfg.dropout),
        )
        self.motif_aware_classifier = nn.Sequential(
            nn.Linear(256 + 256 + 128, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(64, 1),
        )
        self.motif_feature_dim = motif_dim
        self.motif_group_classifier = nn.Sequential(
            nn.Linear(motif_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(64, 2),
        )
        self.motif_conflict_classifier = nn.Sequential(
            nn.Linear(motif_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(inplace=True),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(32, 1),
        )
        self.motif_rescue_latent_head = nn.Sequential(
            nn.Linear(motif_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(128, self.cfg.motif_latent_dim),
        )
        self.motif_group_latent_prototypes = nn.Parameter(torch.zeros(2, self.cfg.motif_latent_dim))
        self._init_rescue_branch()

    def _init_rescue_branch(self):
        for module in [self.motif_group_classifier, self.motif_conflict_classifier, self.motif_rescue_latent_head]:
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.zeros_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.motif_group_latent_prototypes)

    def forward_motif(self, sequences: Sequence[str], phys_feats: torch.Tensor, king_labels: torch.Tensor):
        reconstructed_z, coeffs, attn = self.motif_model(list(sequences), phys_feats, king_labels)
        motif_features = torch.cat(
            [
                coeffs,
                self.motif_model.last_motif_subtype_logits,
                self.motif_model.last_seed_group_logits,
            ],
            dim=1,
        )
        group_logits = self.motif_group_classifier(motif_features)
        conflict_logit = self.motif_conflict_classifier(motif_features).squeeze(1)
        group_probs = torch.sigmoid(group_logits)
        conflict_prob = torch.sigmoid(conflict_logit).unsqueeze(1)
        latent_delta = self.motif_rescue_latent_head(motif_features)
        prototype_delta = torch.matmul(group_probs, self.motif_group_latent_prototypes)
        rescued_coeffs = coeffs + conflict_prob * (latent_delta + prototype_delta)
        rescued_z = torch.matmul(rescued_coeffs, self.motif_model.svd_v) + self.motif_model.svd_mean
        return {
            "reconstructed_z": reconstructed_z,
            "rescued_z": rescued_z,
            "motif_latent": coeffs,
            "rescued_motif_latent": rescued_coeffs,
            "motif_features": motif_features,
            "motif_group_logits": group_logits,
            "motif_conflict_logit": conflict_logit,
            "motif_subtype_logits": self.motif_model.last_motif_subtype_logits,
            "motif_seed_logits": self.motif_model.last_seed_group_logits,
            "attention": attn,
        }

    def forward_binding_base(self, rna_codes: torch.Tensor, protein_rows: torch.Tensor) -> torch.Tensor:
        return self.binding_model(rna_codes, protein_rows)

    def forward_binding_motif_aware(self, rna_codes: torch.Tensor, protein_rows: torch.Tensor, motif_features: torch.Tensor) -> torch.Tensor:
        rna_vec = self.binding_model.encode_rna(rna_codes)
        protein_vectors = self.binding_model.protein_embedding(protein_rows)
        protein_vec = self.binding_model.protein_projector(protein_vectors)
        motif_vec = self.motif_binding_projector(motif_features)
        return self.motif_aware_classifier(torch.cat([rna_vec, protein_vec, motif_vec], dim=1)).squeeze(1)

    def trainable_parameter_groups(self):
        return {
            "motif_model_non_esm": [p for n, p in self.motif_model.named_parameters() if p.requires_grad and not n.startswith("esm2.")],
            "binding_model": [p for p in self.binding_model.parameters() if p.requires_grad],
            "motif_binding_projector": list(self.motif_binding_projector.parameters()),
            "motif_aware_classifier": list(self.motif_aware_classifier.parameters()),
            "motif_rescue_branch": list(self.motif_group_classifier.parameters())
            + list(self.motif_conflict_classifier.parameters())
            + list(self.motif_rescue_latent_head.parameters())
            + [self.motif_group_latent_prototypes],
        }
