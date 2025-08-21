from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import os
import random
import numpy as np
import torch
from contextlib import nullcontext
from transformers import AutoModelForCausalLM, AutoTokenizer


def _pick_dtype(dtype: str | None) -> torch.dtype | str:
    """
    dtype = "auto" | "bfloat16" | "float16" | "float32" | None
    For MXFP4-quantized weights, leave as "auto". On non-Hopper GPUs,
    prefer bfloat16 if you have VRAM; else float16/float32.
    """
    if not dtype or dtype == "auto":
        return "auto"
    return getattr(torch, dtype)


def _seed_everything(seed: int, deterministic: bool = False) -> None:
    """
    Seed Python, NumPy, and PyTorch. Optionally enable deterministic algorithms.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # May reduce performance and disable some fast kernels.
        # Note: for full determinism with CUDA, set CUBLAS_WORKSPACE_CONFIG
        # BEFORE CUDA context is created (ideally at process start).
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _seeded_generation_ctx(seed: Optional[int], device: torch.device | str) -> Any:
    """
    Fork RNG state so per-call seeding doesn't leak globally.
    Works on CPU-only and CUDA. Use around `model.generate(...)`.
    """
    if seed is None:
        return nullcontext()
    # Choose CUDA device(s) to fork if applicable
    devs = None
    dev_str = str(device)
    if torch.cuda.is_available() and "cuda" in dev_str:
        if isinstance(device, torch.device) and device.index is not None:
            devs = [device.index]
        else:
            devs = [torch.cuda.current_device()]
    return torch.random.fork_rng(devices=devs, enabled=True)


def _apply_chat_template(tokenizer, messages: List[Dict[str, str]], model_device) -> Dict[str, Any]:
    """
    Encode OpenAI-style messages using the tokenizer's chat template (Harmony-compatible).
    Falls back to a simple join if template is missing.
    """
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(model_device)

    # Fallback (plain text)
    joined = []
    for m in messages:
        role = m.get("role", "user").upper()
        joined.append(f"{role}: {m['content']}")
    joined.append("ASSISTANT:")
    text = "\n".join(joined)
    return tokenizer(text, return_tensors="pt").to(model_device)


@dataclass
class GPTOSS:
    model: Any
    tokenizer: Any
    model_name_or_path: str
    seed: Optional[int] = None  # default seed for this instance (can be overridden per-call)

    @torch.inference_mode()
    def chat(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 256,
        temperature: float = 0.2,
        top_p: float = 0.95,
        do_sample: Optional[bool] = None,
        seed: Optional[int] = None,
        **gen_kwargs,
    ) -> str:
        """
        Simple chat; uses chat template if available.
        Set `seed` to make sampling reproducible (falls back to self.seed).
        """
        inputs = _apply_chat_template(self.tokenizer, messages, self.model.device)
        seed_to_use = seed if seed is not None else self.seed

        with _seeded_generation_ctx(seed_to_use, self.model.device):
            if seed_to_use is not None:
                torch.manual_seed(seed_to_use)
                if torch.cuda.is_available() and "cuda" in str(self.model.device):
                    torch.cuda.manual_seed_all(seed_to_use)

            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=(temperature > 0.0 if do_sample is None else do_sample),
                return_dict_in_generate=True,
                output_scores=False,
                **gen_kwargs,
            )

        # Decode only newly generated tokens
        start = inputs["input_ids"].shape[-1]
        return self.tokenizer.decode(out.sequences[0][start:], skip_special_tokens=True)

    @torch.inference_mode()
    def generate_raw(
        self,
        input_ids: torch.LongTensor,
        seed: Optional[int] = None,
        **gen_kwargs,
    ) -> Dict[str, Any]:
        """
        Lower-level generate: pass raw token IDs if you prepare inputs yourself
        (e.g., Harmony prefill IDs).
        """
        seed_to_use = seed if seed is not None else self.seed
        with _seeded_generation_ctx(seed_to_use, self.model.device):
            if seed_to_use is not None:
                torch.manual_seed(seed_to_use)
                if torch.cuda.is_available() and "cuda" in str(self.model.device):
                    torch.cuda.manual_seed_all(seed_to_use)

            return self.model.generate(
                input_ids=input_ids.to(self.model.device),
                return_dict_in_generate=True,
                **gen_kwargs,
            )

    @torch.inference_mode()
    def forward_inspect(
        self,
        messages: List[Dict[str, str]],
        output_hidden_states: bool = True,
        output_attentions: bool = True,
        output_router_logits: bool = True,
        generate: bool = True,
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        top_p: float = 0.9,
        return_intermediates: bool = True,
        seed: Optional[int] = None,
        **forward_kwargs,
    ) -> Dict[str, Any]:
        """
        Forward pass with generation to expose model internals and response.

        Reproducibility: pass a `seed` (or set self.seed on the instance).
        We fork RNG and reseed inside the context so results are deterministic
        with respect to the seed without polluting global RNG state.
        """
        inputs = _apply_chat_template(self.tokenizer, messages, self.model.device)
        input_length = inputs["input_ids"].shape[-1]

        if not generate:
            # Single forward pass (deterministic in eval mode)
            outputs = self.model(
                **inputs,
                use_cache=False,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
                output_router_logits=output_router_logits,
                return_dict=True,
                **forward_kwargs,
            )
            result = self._extract_outputs(outputs, inputs)
            result["input_tokens"] = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            return result
        else:
            # Generation with intermediate capture
            result = {
                "input_ids": inputs["input_ids"],
                "input_tokens": self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]),
                "generation_intermediates": [] if return_intermediates else None
            }

            seed_to_use = seed if seed is not None else self.seed
            with _seeded_generation_ctx(seed_to_use, self.model.device):
                if seed_to_use is not None:
                    torch.manual_seed(seed_to_use)
                    if torch.cuda.is_available() and "cuda" in str(self.model.device):
                        torch.cuda.manual_seed_all(seed_to_use)

                gen_outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=temperature > 0.0,
                    return_dict_in_generate=True,
                    output_hidden_states=output_hidden_states,
                    output_attentions=output_attentions,
                    output_scores=True,  # Token-level logits
                    output_router_logits=output_router_logits,
                    **forward_kwargs,
                )

            # Extract generated text
            generated_ids = gen_outputs.sequences[0][input_length:]
            result["generated_ids"] = generated_ids
            result["generated_tokens"] = self.tokenizer.convert_ids_to_tokens(generated_ids)
            result["response"] = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

            # Extract generation internals if available
            if hasattr(gen_outputs, "hidden_states") and gen_outputs.hidden_states:
                result["hidden_states"] = gen_outputs.hidden_states
            if hasattr(gen_outputs, "attentions") and gen_outputs.attentions:
                result["attentions"] = gen_outputs.attentions
            if hasattr(gen_outputs, "scores") and gen_outputs.scores:
                result["scores"] = gen_outputs.scores  # Per-token logits

            # Token-level analysis
            if return_intermediates and hasattr(gen_outputs, "scores"):
                for i, token_scores in enumerate(gen_outputs.scores):
                    token_id = generated_ids[i].item()
                    token_str = self.tokenizer.decode([token_id])
                    probs = torch.softmax(token_scores[0], dim=-1)
                    top_vals, top_idx = torch.topk(probs, k=10)

                    intermediate = {
                        "step": i,
                        "token_id": token_id,
                        "token": token_str,
                        "probability": float(probs[token_id].item()),
                        "top_k_ids": top_idx.tolist(),
                        "top_k_tokens": [self.tokenizer.decode([tid]) for tid in top_idx],
                        "top_k_probs": top_vals.tolist(),
                        "entropy": float(-(probs * torch.log(probs + 1e-12)).sum().item()),
                    }
                    result["generation_intermediates"].append(intermediate)

            return result

    def _extract_outputs(self, outputs, inputs) -> Dict[str, Any]:
        """Extract available outputs from model forward pass."""
        result = {}

        if hasattr(outputs, "logits") and outputs.logits is not None:
            result["logits"] = outputs.logits
            last_token_logits = outputs.logits[0, -1, :]
            last_token_probs = torch.softmax(last_token_logits, dim=-1)
            result["last_token_probs"] = last_token_probs
            result["last_token_entropy"] = -(last_token_probs * torch.log(last_token_probs + 1e-12)).sum()

            top_k = torch.topk(last_token_probs, k=10)
            result["next_token_predictions"] = {
                "token_ids": top_k.indices,
                "tokens": [self.tokenizer.decode([tid]) for tid in top_k.indices],
                "probabilities": top_k.values
            }

        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            result["last_hidden_state"] = outputs.last_hidden_state
        elif hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            result["last_hidden_state"] = outputs.hidden_states[-1]

        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            result["hidden_states"] = outputs.hidden_states

        if hasattr(outputs, "attentions") and outputs.attentions is not None:
            result["attentions"] = outputs.attentions

        if hasattr(outputs, "router_logits") and outputs.router_logits is not None:
            result["router_logits"] = outputs.router_logits
        elif hasattr(outputs, "aux_loss") and outputs.aux_loss is not None:
            result["aux_loss"] = outputs.aux_loss

        if hasattr(outputs, "expert_indices") and outputs.expert_indices is not None:
            result["expert_indices"] = outputs.expert_indices

        return result


def load_gptoss(
    model_name_or_path: str = "openai/gpt-oss-20b",
    *,
    dtype: str | None = "auto",
    device_map: str | Dict[str, int] | None = "auto",
    attn_implementation: Optional[str] = None,
    trust_remote_code: bool = True,
    seed: Optional[int] = None,
    deterministic: bool = False,
    output_router_logits = False,
    **model_kwargs,
) -> GPTOSS:
    """
    Loader tuned for GPT-OSS.

    For strict reproducibility, set `seed` (and optionally `deterministic=True`),
    and consider `attn_implementation="eager"` if fused/flash attention kernels
    are slightly non-deterministic on your stack.
    """
    if seed is not None:
        _seed_everything(seed, deterministic=deterministic)

    tok = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    kwargs = dict(
        torch_dtype=_pick_dtype(dtype),
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    )
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    kwargs.update(model_kwargs)

    mdl = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
    mdl.eval()  # ensure dropout etc. are off

    return GPTOSS(model=mdl, tokenizer=tok, model_name_or_path=model_name_or_path, seed=seed, 
                  output_router_logits=output_router_logits)
