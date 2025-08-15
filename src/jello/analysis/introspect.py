from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch


@dataclass
class MoESummary:
    """Lightweight summary of MoE routing for a sequence."""
    layer_index: int
    topk_expert_ids: List[torch.Tensor]       # shape: [seq_len, top_k]
    topk_expert_probs: List[torch.Tensor]     # shape: [seq_len, top_k]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer_index": self.layer_index,
            "topk_expert_ids": [t.cpu().tolist() for t in self.topk_expert_ids],
            "topk_expert_probs": [t.cpu().tolist() for t in self.topk_expert_probs],
        }


def summarize_router_logits(router_logits: List[torch.Tensor], top_k: int = 2) -> List[MoESummary]:
    """
    router_logits: list of tensors per layer, each [batch, seq, n_experts]
    Returns top-k expert ids/probs per token for each layer.
    """
    summaries: List[MoESummary] = []
    if not router_logits:
        return summaries

    for li, logits in enumerate(router_logits):
        # Assume batch=1 for interactive probing
        if logits.dim() != 3:
            continue
        probs = torch.softmax(logits[0], dim=-1)        # [seq, n_experts]
        p_vals, p_idx = torch.topk(probs, k=min(top_k, probs.shape[-1]), dim=-1)  # [seq, k]
        summaries.append(MoESummary(layer_index=li, topk_expert_ids=[p_idx], topk_expert_probs=[p_vals]))
    return summaries


def layer_activation_norms(hidden_states: List[torch.Tensor]) -> torch.Tensor:
    """
    Quick per-layer L2 norms across seq: shape [num_layers+1, seq_len]
    (includes embedding output at index 0)
    """
    norms = []
    for hs in hidden_states:  # hs: [batch, seq, hidden]
        norms.append(hs[0].norm(dim=-1))  # [seq]
    return torch.stack(norms)  # [layers+1, seq]


def attention_head_entropies(attentions: List[torch.Tensor]) -> torch.Tensor:
    """
    Per-layer, per-head entropy of attention distributions (batch=1 assumed).
    Returns tensor [num_layers, num_heads, seq_len].
    """
    ents = []
    for layer_attn in attentions:  # [batch, heads, seq, seq]
        a = layer_attn[0] + 1e-12
        ent = -(a * a.log()).sum(dim=-1)  # [heads, seq]
        ents.append(ent)
    return torch.stack(ents)  # [layers, heads, seq]
