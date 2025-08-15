# src/jello/metrics/refusal_dir.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Union, Dict

import torch
import numpy as np
import matplotlib.pyplot as plt

from jello.probes.hidden import collect_hidden_states, select_layer_token


TextLike = Union[str, List[Dict[str, str]]]


@dataclass
class Direction:
    layer_idx: int
    vector: torch.Tensor   # [d], normalized or not
    meta: Dict[str, str]   # provenance


def _pool_rep(h_result, layer_idx: int, pool: str = "last") -> torch.Tensor:
    """
    Returns a [d] representation for a single example. pool in {"last", "mean"}.
    """
    h = h_result.hidden_states[layer_idx][0]  # [T, d]
    if pool == "last":
        return h[-1, :].clone()
    elif pool == "mean":
        return h.mean(dim=0).clone()
    else:
        raise ValueError("pool must be 'last' or 'mean'")


@torch.inference_mode()
def build_direction_diff_means(
    model,
    tokenizer,
    texts: Sequence[TextLike],
    labels: Sequence[int],
    layer_idx: int,
    device: Optional[torch.device] = None,
    pool: str = "last",
    normalize: bool = True,
    meta: Optional[Dict[str, str]] = None,
) -> Direction:
    """
    Compute v = mean(refuse) - mean(comply) at layer_idx, using a simple pooling.
    labels: 1 = refuse, 0 = comply.
    """
    device = device or next(model.parameters()).device
    xs_ref, xs_cmp = [], []
    for text, y in zip(texts, labels):
        res = collect_hidden_states(model, tokenizer, text, device=device)
        rep = _pool_rep(res, layer_idx, pool)  # [d]
        (xs_ref if y == 1 else xs_cmp).append(rep)
    mu_ref = torch.stack(xs_ref, dim=0).mean(dim=0)
    mu_cmp = torch.stack(xs_cmp, dim=0).mean(dim=0)
    v = mu_ref - mu_cmp
    if normalize:
        v = v / (v.norm() + 1e-8)
    return Direction(layer_idx=layer_idx, vector=v, meta=meta or {"method": "diff_means", "pool": pool})


@torch.inference_mode()
def score_projection(
    model,
    tokenizer,
    text_or_messages: TextLike,
    direction: Direction,
    device: Optional[torch.device] = None,
    pool: str = "last",
    normalize_v: bool = True,
) -> float:
    """
    Returns ⟨h, v⟩ for a single example at direction.layer_idx.
    """
    res = collect_hidden_states(model, tokenizer, text_or_messages, device=device)
    h = _pool_rep(res, direction.layer_idx, pool)  # [d]
    v = direction.vector.to(h.dtype)
    if normalize_v:
        v = v / (v.norm() + 1e-8)
    return float(torch.dot(h, v).item())


# ---------- CAA / orthogonalization hooks ----------

class _AddDirectionHook:
    """
    Forward pre-hook that adds alpha*v to the input of a module expecting [B, T, d].
    """
    def __init__(self, v: torch.Tensor, alpha: float):
        self.v = v / (v.norm() + 1e-8)
        self.alpha = alpha

    def __call__(self, module, inputs):
        (x,) = inputs  # [B, T, d]
        return (x + self.alpha * self.v.to(x.dtype),)


class _AblateDirectionHook:
    """
    Forward pre-hook that removes the projection along v:
        x <- x - <x,v>/||v||^2 * v
    """
    def __init__(self, v: torch.Tensor):
        nv = v.norm() + 1e-8
        self.v = v / nv
        self.inv_norm2 = 1.0  # since v normalized

    def __call__(self, module, inputs):
        (x,) = inputs  # [B, T, d]
        coeff = (x * self.v.to(x.dtype)).sum(dim=-1, keepdim=True) * self.inv_norm2
        return (x - coeff * self.v.to(x.dtype),)


def _find_layer_module(model, layer_idx: int, prefer_sub: Optional[str] = None):
    """
    Heuristic to find the i-th Transformer block, then an optional submodule (e.g., 'mlp' or 'self_attn').
    Works for common HF naming schemes: model.layers[i], transformer.h[i], etc.
    """
    # 1) Find any ModuleList of length >= layer_idx+1 that contains blocks
    candidates = []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) > layer_idx:
            # quick check: elements look like blocks (have 'mlp' or 'self_attn' attributes commonly)
            elem = mod[layer_idx]
            if any(hasattr(elem, attr) for attr in ("mlp", "feed_forward", "ffn", "self_attn", "attn", "attention")):
                candidates.append((name, mod))
    if not candidates:
        raise RuntimeError("Could not locate a ModuleList of transformer blocks.")
    # Prefer a list actually named 'layers' or 'h'
    def rank(n): return (0 if n.endswith((".layers", ".h")) or n.endswith("layers") or n.endswith("h") else 1, len(n))
    candidates.sort(key=lambda x: rank(x[0]))
    blocks = candidates[0][1]
    block = blocks[layer_idx]
    if prefer_sub:
        # try common names
        for k in (prefer_sub, "mlp", "feed_forward", "ffn", "self_attn", "attn", "attention"):
            if hasattr(block, k):
                return getattr(block, k)
    return block


def register_caa_add(
    model,
    direction: Direction,
    alpha: float = 0.8,
    prefer_sub: Optional[str] = "mlp",
):
    """
    Register a forward-pre hook to ADD alpha*v at the chosen layer/submodule.
    Returns the handle; caller must handle.remove() after use.
    """
    target = _find_layer_module(model, direction.layer_idx, prefer_sub)
    hook = _AddDirectionHook(direction.vector.to(next(model.parameters()).device), alpha)
    return target.register_forward_pre_hook(hook)


def register_caa_ablate(
    model,
    direction: Direction,
    prefer_sub: Optional[str] = "mlp",
):
    """
    Register a forward-pre hook to ABLATE (project out) v at the chosen layer/submodule.
    Returns the handle; caller must handle.remove() after use.
    """
    target = _find_layer_module(model, direction.layer_idx, prefer_sub)
    hook = _AblateDirectionHook(direction.vector.to(next(model.parameters()).device))
    return target.register_forward_pre_hook(hook)


# ---------- Visualization ----------

def plot_score_histograms(
    scores: Sequence[float],
    labels: Sequence[int],
    ax: Optional[plt.Axes] = None,
    title: str = "Refusal-direction projection by label",
):
    """
    Two histograms: scores where y=1 (refuse) vs y=0 (comply).
    """
    ax = ax or plt.gca()
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    ax.hist(scores[labels == 1], bins=30, alpha=0.6, label="refuse (1)")
    ax.hist(scores[labels == 0], bins=30, alpha=0.6, label="comply (0)")
    ax.set_xlabel("⟨h, v⟩")
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.legend()


def plot_threshold_curve(
    scores: Sequence[float],
    labels: Sequence[int],
    ax: Optional[plt.Axes] = None,
    title: str = "Threshold sweep (TPR vs FPR)",
):
    """
    Simple ROC-like threshold sweep without sklearn.
    """
    ax = ax or plt.gca()
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    ths = np.quantile(scores, np.linspace(0, 1, 101))
    tpr, fpr = [], []
    P = (labels == 1).sum()
    N = (labels == 0).sum()
    for th in ths:
        pred = (scores >= th)
        tp = int(((pred == 1) & (labels == 1)).sum())
        fp = int(((pred == 1) & (labels == 0)).sum())
        tpr.append(tp / max(P, 1))
        fpr.append(fp / max(N, 1))
    ax.plot(fpr, tpr, marker=".")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.3)
