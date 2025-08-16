from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
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

    @torch.inference_mode()
    def chat(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 256,
        temperature: float = 0.2,
        top_p: float = 0.95,
        do_sample: Optional[bool] = None,
        **gen_kwargs,
    ) -> str:
        """
        Simple chat; uses chat template if available.
        """
        inputs = _apply_chat_template(self.tokenizer, messages, self.model.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=(temperature > 0.0 if do_sample is None else do_sample),
            return_dict_in_generate=True,
            output_scores=False,
        )
        # Decode only newly generated tokens
        start = inputs["input_ids"].shape[-1]
        return self.tokenizer.decode(out.sequences[0][start:], skip_special_tokens=True)

    @torch.inference_mode()
    def generate_raw(
        self,
        input_ids: torch.LongTensor,
        **gen_kwargs,
    ) -> Dict[str, Any]:
        """
        Lower-level generate: pass raw token IDs if you prepare inputs yourself
        (e.g., Harmony prefill IDs).
        """
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
        output_attentions: bool = False,
        output_router_logits: bool = True,
        **forward_kwargs,
    ) -> Dict[str, Any]:
        """
        One forward pass (no generation) to expose internals:
          - hidden_states: per-layer activations
          - attentions: per-layer attn maps (if enabled)
          - router_logits: MoE routing signals (if the architecture supports it)

        Returns a dict with available fields.
        """
        inputs = _apply_chat_template(self.tokenizer, messages, self.model.device)

        outputs = self.model(
            **inputs,
            use_cache=False,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            output_router_logits=output_router_logits,  # many MoE models expose this
            return_dict=True,
            **forward_kwargs,
        )

        result = {}
        
        # Try to get the main output (logits)
        if hasattr(outputs, "logits") and outputs.logits is not None:
            result["logits"] = outputs.logits
        
        # Try to get last hidden state (different attribute names in different models)
        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            result["last_hidden_state"] = outputs.last_hidden_state
        elif hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            # If no last_hidden_state, use the last layer from hidden_states
            result["last_hidden_state"] = outputs.hidden_states[-1]
        
        # Hidden states (all layers)
        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            result["hidden_states"] = outputs.hidden_states
            
        # Attention weights
        if hasattr(outputs, "attentions") and outputs.attentions is not None:
            result["attentions"] = outputs.attentions
            
        # MoE routing info (various possible attribute names)
        if hasattr(outputs, "router_logits") and outputs.router_logits is not None:
            result["router_logits"] = outputs.router_logits
        elif hasattr(outputs, "aux_loss") and outputs.aux_loss is not None:
            # Some models put routing info in aux_loss
            result["aux_loss"] = outputs.aux_loss
            
        # Expert indices
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
    **model_kwargs,
) -> GPTOSS:
    """
    Loader tuned for GPT-OSS.

    - Defaults match the official Cookbook (auto dtype/device; chat template). :contentReference[oaicite:4]{index=4}
    - If you hit MXFP4 issues on non-Hopper GPUs, retry with dtype="bfloat16". :contentReference[oaicite:5]{index=5}
    """
    tok = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    kwargs = dict(
        torch_dtype=_pick_dtype(dtype),
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    )
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation  # e.g., "kernels-community/vllm-flash-attn3" (120b). :contentReference[oaicite:6]{index=6}
    kwargs.update(model_kwargs)

    mdl = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
    return GPTOSS(model=mdl, tokenizer=tok, model_name_or_path=model_name_or_path)
