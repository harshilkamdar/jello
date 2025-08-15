# src/jello/probes/hidden.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import numpy as np
import matplotlib.pyplot as plt


TextLike = Union[str, List[Dict[str, str]]]  # raw text or OpenAI-style messages


@dataclass
class HiddenProbeResult:
    input_ids: torch.Tensor           # [1, T]
    tokens: List[str]                 # length T
    hidden_states: List[torch.Tensor] # L+1 tensors, each [1, T, d]
    layer_token_norms: np.ndarray     # (L+1, T) L2 norm per layer per token


def _apply_chat_template_or_encode(
    tokenizer,
    text_or_messages: TextLike,
    add_generation_prompt: bool = False,
    device: Optional[torch.device] = None,
):
    """
    Returns tokenized inputs for HF models. If messages are passed and the tokenizer
    has a chat template, it will be used; otherwise falls back to plain encode.
    """
    if isinstance(text_or_messages, list) and hasattr(tokenizer, "apply_chat_template"):
        enc = tokenizer.apply_chat_template(
            text_or_messages,
            add_generation_prompt=add_generation_prompt,
            return_tensors="pt",
            return_dict=True,
        )
    else:
        enc = tokenizer(
            text_or_messages if isinstance(text_or_messages, str) else str(text_or_messages),
            return_tensors="pt",
            return_dict=True,
        )
    if device is not None:
        enc = {k: v.to(device) for k, v in enc.items()}
    return enc


def _tokens_from_ids(tokenizer, input_ids: torch.Tensor) -> List[str]:
    ids = input_ids[0].tolist()
    return tokenizer.convert_ids_to_tokens(ids)


def activation_norms(hidden_states: Sequence[torch.Tensor]) -> np.ndarray:
    """
    Compute per-layer per-token L2 norms. hidden_states: list of [1, T, d], length L+1.
    Returns np.array of shape (L+1, T).
    """
    norms = []
    for hs in hidden_states:  # [1, T, d]
        with torch.no_grad():
            n = torch.norm(hs[0], dim=-1).cpu().float().numpy()  # [T]
        norms.append(n)
    return np.stack(norms)  # (L+1, T)


@torch.inference_mode()
def collect_hidden_states(
    model,
    tokenizer,
    text_or_messages: TextLike,
    device: Optional[torch.device] = None,
) -> HiddenProbeResult:
    """
    Runs a single forward pass to capture all hidden states (no generation).

    Returns:
        HiddenProbeResult with tokens, hidden_states, and activation norms.
    """
    device = device or next(model.parameters()).device
    enc = _apply_chat_template_or_encode(tokenizer, text_or_messages, False, device)
    out = model(
        **enc,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    hs = list(out.hidden_states)  # L+1
    toks = _tokens_from_ids(tokenizer, enc["input_ids"])
    norms = activation_norms(hs)
    return HiddenProbeResult(
        input_ids=enc["input_ids"].detach().cpu(),
        tokens=toks,
        hidden_states=[h.detach().cpu() for h in hs],
        layer_token_norms=norms,
    )


def select_layer_token(
    result: HiddenProbeResult, layer_idx: int, token_pos: int = -1
) -> torch.Tensor:
    """
    Get the representation vector at a chosen (layer, token).
    """
    h = result.hidden_states[layer_idx]  # [1, T, d]
    T = h.shape[1]
    pos = token_pos if token_pos >= 0 else (T - 1)
    return h[0, pos, :].clone()  # [d]


def project_direction(
    result: HiddenProbeResult,
    direction: torch.Tensor,
    layer_idx: int,
    normalize_v: bool = True,
) -> np.ndarray:
    """
    Compute ⟨h_{t}^{(layer)}, v⟩ across tokens (returns length-T vector).
    """
    h = result.hidden_states[layer_idx][0]  # [T, d]
    v = direction.to(h.dtype)
    if normalize_v:
        v = v / (v.norm() + 1e-8)
    proj = (h @ v).cpu().float().numpy()  # [T]
    return proj


# ---------- Visualization helpers ----------

def plot_activation_norms(
    result: HiddenProbeResult,
    ax: Optional[plt.Axes] = None,
    title: str = "Activation norms (L2) per layer × token",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
):
    """
    Heatmap of activation norms. y-axis: layer (0..L), x-axis: tokens.
    """
    data = result.layer_token_norms  # (L+1, T)
    ax = ax or plt.gca()
    im = ax.imshow(data, aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax)
    ax.set_ylabel("Layer (0..L)")
    ax.set_xlabel("Token position")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)


def plot_token_projection(
    tokens: List[str],
    projection: np.ndarray,
    ax: Optional[plt.Axes] = None,
    title: str = "Per-token projection ⟨h, v⟩",
):
    """
    Line plot of per-token projection along a direction vector.
    """
    ax = ax or plt.gca()
    ax.plot(np.arange(len(tokens)), projection, marker="o")
    ax.set_xticks(np.arange(len(tokens)))
    # Show compact token labels; truncate very long tokens
    lbls = [t if len(t) < 10 else t[:9] + "…" for t in tokens]
    ax.set_xticklabels(lbls, rotation=90)
    ax.set_ylabel("⟨h, v⟩")
    ax.set_xlabel("Token")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.3)
