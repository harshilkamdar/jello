# src/jello/probes/attention.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import numpy as np
import matplotlib.pyplot as plt

TextLike = Union[str, List[Dict[str, str]]]


@dataclass
class AttentionProbeResult:
    input_ids: torch.Tensor             # [1, T]
    tokens: List[str]                   # length T
    attentions: List[torch.Tensor]      # L tensors, each [H, T, T]
    # Convenience caches:
    num_layers: int
    num_heads: int
    seq_len: int


def _apply_chat_template_or_encode(
    tokenizer, text_or_messages: TextLike, add_generation_prompt: bool = False, device=None
):
    if isinstance(text_or_messages, list) and hasattr(tokenizer, "apply_chat_template"):
        enc = tokenizer.apply_chat_template(
            text_or_messages, add_generation_prompt=add_generation_prompt, return_tensors="pt", return_dict=True
        )
    else:
        enc = tokenizer(text_or_messages if isinstance(text_or_messages, str) else str(text_or_messages),
                        return_tensors="pt", return_dict=True)
    if device is not None:
        enc = {k: v.to(device) for k, v in enc.items()}
    return enc


def _tokens_from_ids(tokenizer, input_ids: torch.Tensor) -> List[str]:
    return tokenizer.convert_ids_to_tokens(input_ids[0].tolist())


@torch.inference_mode()
def collect_attentions(
    model,
    tokenizer,
    text_or_messages: TextLike,
    device: Optional[torch.device] = None,
) -> AttentionProbeResult:
    """
    Forward pass with output_attentions=True and no generation.
    Returns layer-wise attentions with batch squeezed.
    """
    device = device or next(model.parameters()).device
    enc = _apply_chat_template_or_encode(tokenizer, text_or_messages, False, device)
    out = model(
        **enc,
        output_attentions=True,
        output_hidden_states=False,
        use_cache=False,
        return_dict=True,
    )
    if out.attentions is None:
        raise RuntimeError("Model did not return attentions. Ensure it supports output_attentions=True.")
    # out.attentions: tuple of L tensors [B, H, T, T]
    atts = [a[0].detach().cpu() for a in out.attentions]  # strip batch -> [H, T, T]
    tokens = _tokens_from_ids(tokenizer, enc["input_ids"])
    L = len(atts)
    H, T, _ = atts[0].shape
    return AttentionProbeResult(
        input_ids=enc["input_ids"].detach().cpu(),
        tokens=tokens,
        attentions=atts,
        num_layers=L,
        num_heads=H,
        seq_len=T,
    )


def head_distraction_index(
    attn: AttentionProbeResult,
    instruction_span: Tuple[int, int],
    injection_span: Tuple[int, int],
    query_pos: int = -1,
) -> np.ndarray:
    """
    For each layer and head, compute:
        D[l,h] = sum_{s in injection} A[l,h,query_pos,s] - sum_{s in instruction} A[l,h,query_pos,s]
    Positive D => more mass to injection than to instruction (distraction).
    Returns np.array shape (L, H).
    """
    q = query_pos if query_pos >= 0 else (attn.seq_len - 1)
    i0, i1 = instruction_span
    j0, j1 = injection_span
    D = np.zeros((attn.num_layers, attn.num_heads), dtype=np.float32)
    for li, A in enumerate(attn.attentions):
        # A: [H, T, T]
        instr_mass = A[:, q, i0:i1].sum(dim=-1)  # [H]
        inj_mass = A[:, q, j0:j1].sum(dim=-1)    # [H]
        d = (inj_mass - instr_mass).cpu().numpy()
        D[li, :] = d
    return D


# ---------- Visualization helpers ----------

def plot_attention_heatmap(
    attn: AttentionProbeResult,
    layer: int,
    head: int,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    segments: Optional[Dict[str, Tuple[int, int]]] = None,
):
    """
    Heatmap of A[layer, head] (T × T). Optionally overlays vertical spans for segments.
    """
    ax = ax or plt.gca()
    A = attn.attentions[layer][head].numpy()  # [T, T]
    im = ax.imshow(A, aspect="auto", interpolation="nearest")
    ax.set_xlabel("Key positions (s)")
    ax.set_ylabel("Query positions (t)")
    ttl = title or f"Attention L{layer} H{head}"
    ax.set_title(ttl)
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    if segments:
        for name, (s0, s1) in segments.items():
            ax.axvspan(s0, s1 - 1, alpha=0.15, label=name)
        ax.legend(loc="upper right")


def plot_head_distraction_bars(
    D: np.ndarray,
    layer: int,
    ax: Optional[plt.Axes] = None,
    title: str = "Head-Distraction index (inj - instr)",
):
    """
    Bar plot of distraction per head at a chosen layer. D has shape (L, H).
    """
    ax = ax or plt.gca()
    vals = D[layer]
    ax.bar(np.arange(len(vals)), vals)
    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_xlabel("Head")
    ax.set_ylabel("Δ attention mass")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.3)


# ---------- Optional: try head ablation via head_mask ----------

@torch.inference_mode()
def run_with_head_mask(
    model,
    tokenizer,
    text_or_messages: TextLike,
    head_mask: torch.Tensor,  # [L, H] with 0 to ablate, 1 to keep
    device: Optional[torch.device] = None,
):
    """
    Attempts to call the model with a head_mask (supported by many HF attention modules, but not all).
    If the model doesn't support it, this will raise or be ignored by forward().
    """
    device = device or next(model.parameters()).device
    enc = _apply_chat_template_or_encode(tokenizer, text_or_messages, False, device)
    try:
        out = model(
            **enc,
            head_mask=head_mask.to(device),
            output_attentions=False,
            use_cache=False,
            return_dict=True,
        )
        return out
    except TypeError:
        raise RuntimeError("This model.forward does not accept head_mask. Consider module-level hooks instead.")
