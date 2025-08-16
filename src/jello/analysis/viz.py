from __future__ import annotations
import os
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import matplotlib.pyplot as plt

# --------------------------- Helpers --------------------------------

def _ensure_out_dir(out_dir: Optional[str]) -> None:
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

def _to_numpy(x: Union[torch.Tensor, np.ndarray, float]) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().to("cpu").float().numpy()
    if isinstance(x, (float, int)):
        return np.array([float(x)])
    return x

def _logsumexp_torch(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    m, _ = torch.max(logits, dim=dim, keepdim=True)
    z = m + torch.log(torch.sum(torch.exp(logits - m), dim=dim, keepdim=True))
    return z.squeeze(dim)

def _safe_decode_token(tokenizer, token_id: int) -> str:
    try:
        return tokenizer.decode([token_id]).replace("\n", "\\n")
    except Exception:
        return f"<{token_id}>"

def _standardize_scores(scores: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    """
    HF generate(..., return_dict_in_generate=True, output_scores=True) yields a list
    of length T where each tensor is [batch, vocab]. We assume batch=1.
    """
    out = []
    for t in scores:
        if not isinstance(t, torch.Tensor):
            raise TypeError("scores contains non-tensors; unexpected shape.")
        if t.ndim == 2:  # [B,V]
            out.append(t[0])
        else:
            # Some models could return [V] already
            out.append(t.view(-1))
    return out  # list of [V] tensors (logits)

def _standardize_generated_ids(res: Dict[str, Any]) -> torch.LongTensor:
    gids = res.get("generated_ids", None)
    if gids is None:
        seq = res["sequences"][0] if "sequences" in res else None
        input_len = res["input_ids"].shape[-1]
        if seq is None:
            raise ValueError("No generated_ids or sequences in result dict.")
        return seq[input_len:]
    if isinstance(gids, torch.Tensor):
        return gids
    return torch.tensor(gids)

def _is_per_step_nested(nested: Any) -> bool:
    """
    Heuristic to detect if the top-level object is 'per-generation-step'.
    For HF generate(..., output_attentions/hidden_states=True), top-level is a tuple/list of length T;
    each element is a tuple per layer.
    """
    if isinstance(nested, (list, tuple)) and len(nested) > 0:
        return isinstance(nested[0], (list, tuple))
    return False

def _as_list(x):
    return x if isinstance(x, (list, tuple)) else [x]

# --------------------------- Token confidence / entropy --------------

def _selected_token_probabilities(res: Dict[str, Any]) -> np.ndarray:
    """
    Returns array shape [T] with the probability assigned to the actually generated token at each step.
    """
    scores = _standardize_scores(res["scores"])
    gids = _standardize_generated_ids(res)

    probs = []
    for t, logits in enumerate(scores):
        token_id = int(gids[t].item())
        # compute log-softmax on logits once and re-use
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        probs.append(float(torch.exp(log_probs[token_id]).item()))
    return np.array(probs)

def _step_entropies(res: Dict[str, Any]) -> np.ndarray:
    """
    Compute H(p_t) for each step t where p_t = softmax(scores[t]).
    Entropy H = -sum_i p_i log p_i.
    """
    scores = _standardize_scores(res["scores"])
    ent = []
    for logits in scores:
        logp = torch.nn.functional.log_softmax(logits, dim=-1)
        p = torch.exp(logp)
        ent.append(float(-(p * logp).sum().item()))
    return np.array(ent)

def plot_token_confidence_timeline(res: Dict[str, Any], out_dir: Optional[str] = None) -> None:
    """
    Plot probability assigned to the chosen token at each generation step.
    """
    _ensure_out_dir(out_dir)
    probs = _selected_token_probabilities(res)
    plt.figure()
    plt.plot(np.arange(1, len(probs)+1), probs, marker="o")
    plt.xlabel("Generation step")
    plt.ylabel("P(chosen token)")
    plt.title("Token confidence over generation")
    if out_dir:
        plt.savefig(os.path.join(out_dir, "token_confidence_timeline.png"), dpi=150, bbox_inches="tight")
    plt.show()

def plot_entropy_timeline(res: Dict[str, Any], out_dir: Optional[str] = None) -> None:
    """
    Plot predictive entropy per step (captures model uncertainty).
    """
    _ensure_out_dir(out_dir)
    ent = _step_entropies(res)
    plt.figure()
    plt.plot(np.arange(1, len(ent)+1), ent, marker="o")
    plt.xlabel("Generation step")
    plt.ylabel("Entropy (nats)")
    plt.title("Predictive entropy over generation")
    if out_dir:
        plt.savefig(os.path.join(out_dir, "entropy_timeline.png"), dpi=150, bbox_inches="tight")
    plt.show()

def plot_topk_probs_final_step(res: Dict[str, Any], k: int = 10, out_dir: Optional[str] = None, tokenizer=None) -> None:
    """
    Show top-k probabilities at the final step (useful "waterfall" view of what model almost chose).
    """
    _ensure_out_dir(out_dir)
    scores = _standardize_scores(res["scores"])
    last_logits = scores[-1]
    topk = torch.topk(torch.nn.functional.softmax(last_logits, dim=-1), k=k)
    tokens = []
    if tokenizer is not None:
        tokens = [tokenizer.decode([int(i)]) for i in topk.indices.tolist()]
    else:
        tokens = [str(int(i)) for i in topk.indices.tolist()]

    plt.figure()
    xs = np.arange(k)
    plt.bar(xs, _to_numpy(topk.values))
    plt.xticks(xs, [t.replace("\n","\\n") for t in tokens], rotation=45, ha="right")
    plt.xlabel("Token")
    plt.ylabel("Probability")
    plt.title(f"Top-{k} probs at final step")
    if out_dir:
        plt.savefig(os.path.join(out_dir, f"top{k}_final_step.png"), dpi=150, bbox_inches="tight")
    plt.show()

# --------------------------- Attention utilities --------------------

def standardize_attentions(attentions: Any) -> torch.Tensor:
    """
    Standardize attention returns from HF generate(...) or forward(...).
    Output shape: [T, L, H, S, S], where:
        T = number of generation steps (T=1 if forward-only)
        L = layers
        H = heads
        S = sequence length at that step
    """
    if attentions is None:
        raise ValueError("No attentions available.")
    # Case A: per-step nested -> len = T; each item = tuple of L tensors [B,H,S,S]
    if _is_per_step_nested(attentions):
        per_step = []
        for step_pack in attentions:
            step_layers = []
            for A in step_pack:
                if not isinstance(A, torch.Tensor):
                    raise TypeError("Attention entry is not a tensor.")
                # Expected [B,H,S,S]
                if A.ndim == 4:
                    A = A
                elif A.ndim == 3:
                    # If [H,S,S] (no batch), add batch dim
                    A = A.unsqueeze(0)
                else:
                    raise ValueError(f"Unexpected attention tensor shape: {tuple(A.shape)}")
                step_layers.append(A[0])  # [H,S,S]
            per_step.append(torch.stack(step_layers, dim=0))  # [L,H,S,S]
        return torch.stack(per_step, dim=0)  # [T,L,H,S,S]
    # Case B: single forward pack -> tuple of L tensors [B,H,S,S]
    elif isinstance(attentions, (list, tuple)) and len(attentions) > 0 and isinstance(attentions[0], torch.Tensor):
        step_layers = []
        for A in attentions:
            if A.ndim == 4:
                A = A
            elif A.ndim == 3:
                A = A.unsqueeze(0)
            else:
                raise ValueError(f"Unexpected attention tensor shape: {tuple(A.shape)}")
            step_layers.append(A[0])  # [H,S,S]
        one = torch.stack(step_layers, dim=0).unsqueeze(0)  # [1,L,H,S,S]
        return one
    else:
        raise TypeError("Unsupported attentions structure.")

def _attention_head_entropy(A_lastpos: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Entropy for each head across source positions at the last query position.
    A_lastpos: [L,H,S] attention probs (already normalized) for the last token (query pos = -1).
    Return: [L,H] entropies.
    """
    P = torch.clamp(A_lastpos, eps, 1.0)  # avoid log(0)
    H = -(P * torch.log(P)).sum(dim=-1)   # sum over src positions
    return H  # [L,H]

def plot_attention_entropy_heatmap(res: Dict[str, Any], step: int = -1, out_dir: Optional[str] = None) -> None:
    """
    Heatmap of attention head entropy at the chosen step (lower entropy = peakier, more focused heads).
    """
    _ensure_out_dir(out_dir)
    A = standardize_attentions(res["attentions"])  # [T,L,H,S,S]
    T, L, H, S, _ = A.shape
    step = (T + step) % T
    A_step = A[step]   # [L,H,S,S]
    # last query position (new token)
    A_last = A_step[:, :, -1, :]  # [L,H,S]
    ent = _attention_head_entropy(A_last)  # [L,H]
    fig = plt.figure()
    plt.imshow(_to_numpy(ent), aspect="auto")
    plt.colorbar(label="Entropy (nats)")
    plt.xlabel("Head")
    plt.ylabel("Layer")
    plt.title(f"Attention head entropy (step {step+1}/{T})")
    if out_dir:
        plt.savefig(os.path.join(out_dir, f"attention_entropy_step{step+1}.png"), dpi=150, bbox_inches="tight")
    plt.show()

def plot_attention_to_prompt_tokens(
    res: Dict[str, Any],
    tokenizer,
    layer: Union[int, str] = "best_by_entropy",
    step: int = -1,
    topn: int = 30,
    out_dir: Optional[str] = None,
) -> None:
    """
    Show which prompt tokens the *last generated token* attends to, averaged across heads for one layer.
    layer: int index or "best_by_entropy" to auto-pick the lowest-entropy layer for the last step.
    """
    _ensure_out_dir(out_dir)
    A = standardize_attentions(res["attentions"])  # [T,L,H,S,S]
    toks = res.get("input_tokens", None)
    if toks is None:
        # fallback: decode from input_ids
        toks = [tokenizer.decode([int(t)]) for t in res["input_ids"][0].tolist()]
    T, L, H, S, _ = A.shape
    step = (T + step) % T
    ent_per_layer = torch.empty(L)
    A_step = A[step]  # [L,H,S,S]
    A_last = A_step[:, :, -1, :]  # [L,H,S]
    for li in range(L):
        ent_per_layer[li] = _attention_head_entropy(A_last[li:li+1]).mean()
    if isinstance(layer, str) and layer == "best_by_entropy":
        layer_idx = int(torch.argmin(ent_per_layer).item())
    else:
        layer_idx = int(layer)
    contrib = A_last[layer_idx].mean(dim=0)  # [S] avg over heads
    # Top contributors among prompt positions only (exclude the final position)
    s_prompt = S - 1
    contrib_prompt = contrib[:s_prompt]
    vals, idxs = torch.topk(contrib_prompt, k=min(topn, s_prompt))
    idxs = idxs.tolist()
    labels = [str(i)+":"+(toks[i].replace("\n","\\n")) for i in idxs]
    plt.figure()
    xs = np.arange(len(idxs))
    plt.bar(xs, _to_numpy(vals))
    plt.xticks(xs, labels, rotation=45, ha="right")
    plt.xlabel("Prompt token (index:text)")
    plt.ylabel("Avg attention weight")
    plt.title(f"Layer {layer_idx} attention to prompt tokens (step {step+1}/{T})")
    if out_dir:
        plt.savefig(os.path.join(out_dir, f"attention_prompt_layer{layer_idx}_step{step+1}.png"),
                    dpi=150, bbox_inches="tight")
    plt.show()

def _attention_rollout(A_step: torch.Tensor, alpha: float=0.95) -> torch.Tensor:
    """
    Attention rollout (Abnar & Zuidema, 2020).
    A_step: [L,H,S,S] attention probs for a single step (growing seq).
    Returns vector [S] = contributions of source positions to the final token after rolling through layers.
    """
    L, H, S, _ = A_step.shape
    # Average heads per layer
    A_bar = A_step.mean(dim=1)  # [L,S,S]
    # Residual connection via I blend
    I = torch.eye(S, device=A_bar.device, dtype=A_bar.dtype)
    A_tilde = alpha * A_bar + (1.0 - alpha) * I  # [L,S,S]
    M = A_tilde[0]
    for l in range(1, L):
        M = A_tilde[l] @ M
    # Take the final token's distribution over source tokens
    rollout_vec = M[-1]  # [S]
    return rollout_vec

def plot_attention_rollout_top_tokens(
    res: Dict[str, Any],
    tokenizer,
    alpha: float = 0.95,
    step: int = -1,
    topn: int = 30,
    out_dir: Optional[str] = None,
) -> None:
    """
    Rollout across layers to estimate aggregate influence of prompt tokens on the last token.
    """
    _ensure_out_dir(out_dir)
    A = standardize_attentions(res["attentions"])
    toks = res.get("input_tokens", None)
    if toks is None:
        toks = [tokenizer.decode([int(t)]) for t in res["input_ids"][0].tolist()]
    T, L, H, S, _ = A.shape
    step = (T + step) % T
    A_step = A[step]  # [L,H,S,S]
    rollout = _attention_rollout(A_step, alpha=alpha)  # [S]
    s_prompt = S - 1
    vals, idxs = torch.topk(rollout[:s_prompt], k=min(topn, s_prompt))
    labels = [str(i)+":"+(toks[i].replace("\n","\\n")) for i in idxs.tolist()]
    plt.figure()
    xs = np.arange(len(idxs))
    plt.bar(xs, _to_numpy(vals))
    plt.xticks(xs, labels, rotation=45, ha="right")
    plt.xlabel("Prompt token (index:text)")
    plt.ylabel("Rollout contribution")
    plt.title(f"Attention rollout top tokens (step {step+1}/{T}, alpha={alpha})")
    if out_dir:
        plt.savefig(os.path.join(out_dir, f"attention_rollout_step{step+1}.png"),
                    dpi=150, bbox_inches="tight")
    plt.show()

# --------------------------- Residual stream & Logit Lens -----------

def _standardize_hidden_states(hidden_states: Any) -> List[List[torch.Tensor]]:
    """
    Standardize hidden states into a per-step list:
      return: hs[step][layer] = tensor [B,S,D], where 'layer' runs 0..L (including embedding)
    HF generate(..., output_hidden_states=True) usually returns a tuple length T;
    each element is a tuple of (L+1) tensors [B,S,D].
    If it's a single forward pass, we create T=1.
    """
    if hidden_states is None:
        raise ValueError("No hidden_states available.")
    out = []
    if _is_per_step_nested(hidden_states):
        for step_pack in hidden_states:  # tuple length L+1
            layers = []
            for h in step_pack:
                if not isinstance(h, torch.Tensor):
                    raise TypeError("Hidden state entry is not a tensor.")
                if h.ndim != 3:
                    raise ValueError(f"Unexpected hidden state shape: {tuple(h.shape)}")
                layers.append(h)  # [B,S,D]
            out.append(layers)
        return out
    elif isinstance(hidden_states, (list, tuple)) and len(hidden_states) > 0 and isinstance(hidden_states[0], torch.Tensor):
        # Single forward
        layers = [h for h in hidden_states]
        return [layers]
    else:
        raise TypeError("Unsupported hidden_states structure.")

def plot_residual_norms_heatmap(res: Dict[str, Any], out_dir: Optional[str] = None) -> None:
    """
    Heatmap of ||h_l||_2 at the last position, for each layer l and generation step t.
    """
    _ensure_out_dir(out_dir)
    hs = _standardize_hidden_states(res["hidden_states"])  # list[T][L+1][B,S,D]
    T = len(hs)
    Lp1 = len(hs[0])    # includes embedding
    norms = np.zeros((Lp1, T), dtype=float)
    for t in range(T):
        for l in range(Lp1):
            h = hs[t][l][0, -1, :]  # [D]
            norms[l, t] = float(torch.linalg.vector_norm(h).item())
    plt.figure()
    plt.imshow(norms, aspect="auto")
    plt.colorbar(label="L2 norm")
    plt.xlabel("Generation step")
    plt.ylabel("Layer (0=embed)")
    plt.title("Residual stream norms (last position)")
    if out_dir:
        plt.savefig(os.path.join(out_dir, "residual_norms_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.show()

def _maybe_final_norm(model, h: torch.Tensor) -> torch.Tensor:
    """
    Try applying the model's final normalization used before lm_head.
    Handles common attribute names; falls back to identity if unavailable.
    """
    # Ensure 2D [B,S,D]
    need_unsqueeze = False
    if h.ndim == 1:
        h = h.unsqueeze(0).unsqueeze(0)
        need_unsqueeze = True
    elif h.ndim == 2:
        h = h.unsqueeze(0)
        need_unsqueeze = True

    cand_attrs = ["final_layernorm", "ln_f", "norm", "norm_f"]
    for attr in cand_attrs:
        if hasattr(model, attr):
            try:
                out = getattr(model, attr)(h)
                return out if not need_unsqueeze else out[0, 0, :]
            except Exception:
                pass
    # Fallback: identity
    return h[0,0,:] if need_unsqueeze else h

def plot_logit_lens_selected_heatmap(
    res: Dict[str, Any],
    model: Any,
    out_dir: Optional[str] = None,
) -> None:
    """
    For each step t and layer l, project h_l (last position) through the model's output head
    and extract the logit for the actually generated token at step t.
    Produces a heatmap [layers x steps] of "logit-lens to chosen token".
    """
    _ensure_out_dir(out_dir)
    if "hidden_states" not in res:
        raise ValueError("Result dict lacks hidden_states required for logit lens.")
    if not hasattr(model, "lm_head"):
        raise ValueError("Model lacks lm_head; cannot compute logit lens.")

    hs = _standardize_hidden_states(res["hidden_states"])  # list[T][L+1][B,S,D]
    gids = _standardize_generated_ids(res)
    T = len(hs)
    Lp1 = len(hs[0])

    lens = np.zeros((Lp1, T), dtype=float)
    for t in range(T):
        token_id = int(gids[t].item())
        for l in range(Lp1):
            h = hs[t][l][0, -1, :]  # [D]
            h = _maybe_final_norm(model, h)  # [D]
            logits = model.lm_head(h)        # [V]
            lens[l, t] = float(logits[token_id].item())
    plt.figure()
    plt.imshow(lens, aspect="auto")
    plt.colorbar(label="Logit for chosen token")
    plt.xlabel("Generation step")
    plt.ylabel("Layer (0=embed)")
    plt.title("Logit lens for chosen token across layers/steps")
    if out_dir:
        plt.savefig(os.path.join(out_dir, "logit_lens_chosen_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.show()

# --------------------------- MoE routing -----------------------------

def _standardize_router_logits(router_logits: Any) -> Optional[List[List[torch.Tensor]]]:
    """
    Try to standardize router logits structure into per-step per-layer tensors.
    Return None if not available.
    Output: list[T][L] tensors.
    Each tensor is expected to have one of:
      [B,S,E] or [B,G,S,E] or [B,S,G,E]
    We will reduce to last-position [E] per layer by averaging over groups if present.
    """
    if router_logits is None:
        return None
    out = []
    if _is_per_step_nested(router_logits):
        for step_pack in router_logits:
            layers = []
            for r in step_pack:
                if not isinstance(r, torch.Tensor):
                    # Some models wrap router info differently; skip if unusable
                    continue
                layers.append(r)
            out.append(layers)
        return out if len(out) > 0 else None
    elif isinstance(router_logits, (list, tuple)) and len(router_logits) > 0 and isinstance(router_logits[0], torch.Tensor):
        # single forward
        return [list(router_logits)]
    else:
        return None

def _lastpos_router_probs(r: torch.Tensor) -> torch.Tensor:
    """
    Given router logits tensor r with shape in { [B,S,E], [B,G,S,E], [B,S,G,E] },
    return probabilities over experts at the last position: [E] (averaging groups if present).
    """
    if r.ndim == 3:
        # [B,S,E]
        r_last = r[0, -1, :]          # [E]
        return torch.nn.functional.softmax(r_last, dim=-1)
    elif r.ndim == 4:
        # detect whether [B,G,S,E] or [B,S,G,E]
        if r.shape[1] < 32 and r.shape[2] > 2:
            # assume [B,G,S,E]
            r_last = r[0, :, -1, :]   # [G,E]
        else:
            # assume [B,S,G,E]
            r_last = r[0, -1, :, :]   # [G,E]
        r_last = r_last.mean(dim=0)   # [E] average over groups
        return torch.nn.functional.softmax(r_last, dim=-1)
    else:
        raise ValueError(f"Unexpected router logits shape: {tuple(r.shape)}")

def plot_router_entropy_heatmap(res: Dict[str, Any], out_dir: Optional[str] = None) -> None:
    """
    Heatmap of router entropy at last position per layer and step
    (lower entropy => route concentrated on a few experts).
    """
    _ensure_out_dir(out_dir)
    router = _standardize_router_logits(res.get("router_logits", None))
    if router is None:
        raise ValueError("router_logits not available in result dict.")
    T = len(router)
    L = len(router[0])
    Hm = np.zeros((L, T), dtype=float)
    for t in range(T):
        for l in range(L):
            probs = _lastpos_router_probs(router[t][l])  # [E]
            Hm[l, t] = float(-(probs * torch.log(torch.clamp(probs, 1e-12, 1.0))).sum().item())
    plt.figure()
    plt.imshow(Hm, aspect="auto")
    plt.colorbar(label="Router entropy (nats)")
    plt.xlabel("Generation step")
    plt.ylabel("Layer")
    plt.title("MoE router entropy across layers/steps")
    if out_dir:
        plt.savefig(os.path.join(out_dir, "router_entropy_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.show()

def plot_expert_usage_histogram(res: Dict[str, Any], out_dir: Optional[str] = None, topn: Optional[int] = None) -> None:
    """
    Aggregate top-1 expert selections across layers/steps and show a histogram.
    Useful for spotting imbalance/collapse.
    """
    _ensure_out_dir(out_dir)
    router = _standardize_router_logits(res.get("router_logits", None))
    if router is None:
        raise ValueError("router_logits not available in result dict.")
    counts = {}
    for t in range(len(router)):
        for l in range(len(router[t])):
            probs = _lastpos_router_probs(router[t][l])  # [E]
            top1 = int(torch.argmax(probs).item())
            counts[top1] = counts.get(top1, 0) + 1
    items = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    if topn is not None:
        items = items[:topn]
    labels = [str(k) for k, _ in items]
    vals = [v for _, v in items]
    plt.figure()
    xs = np.arange(len(labels))
    plt.bar(xs, vals)
    plt.xticks(xs, labels, rotation=45, ha="right")
    plt.xlabel("Expert id")
    plt.ylabel("Selections (top-1 across L,T)")
    plt.title("Expert usage histogram")
    if out_dir:
        plt.savefig(os.path.join(out_dir, "expert_usage_histogram.png"), dpi=150, bbox_inches="tight")
    plt.show()