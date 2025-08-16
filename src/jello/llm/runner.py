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
        output_attentions: bool = True,
        output_router_logits: bool = True,
        generate: bool = True,
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        top_p: float = 0.9,
        return_intermediates: bool = True,
        **forward_kwargs,
    ) -> Dict[str, Any]:
        """
        Forward pass with generation to expose model internals and response.
        
        Args:
            messages: Chat messages
            output_hidden_states: Capture per-layer activations
            output_attentions: Capture attention weights
            output_router_logits: Capture MoE routing (if available)
            generate: If True, run generation and capture intermediates
            max_new_tokens: Tokens to generate (512 fits well on RTX 5090)
            temperature: Sampling temperature (1.0 for diverse outputs)
            top_p: Nucleus sampling parameter
            return_intermediates: Capture per-token generation details
            
        Returns:
            Dict with model internals, response, and per-token analysis.
        """
        inputs = _apply_chat_template(self.tokenizer, messages, self.model.device)
        input_length = inputs["input_ids"].shape[-1]
        
        if not generate:
            # Single forward pass (original behavior)
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
            
            # Generate with internals capture
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
                    
                    intermediate = {
                        "step": i,
                        "token_id": token_id,
                        "token": token_str,
                        "probability": probs[token_id].item(),
                        "top_k_probs": torch.topk(probs, k=10),
                        "entropy": -(probs * torch.log(probs + 1e-12)).sum().item(),
                    }
                    result["generation_intermediates"].append(intermediate)
            
            return result

    def _extract_outputs(self, outputs, inputs) -> Dict[str, Any]:
        """Extract available outputs from model forward pass."""
        result = {}
        
        # Try to get the main output (logits)
        if hasattr(outputs, "logits") and outputs.logits is not None:
            result["logits"] = outputs.logits
            
            # Add probability analysis
            last_token_logits = outputs.logits[0, -1, :]  # Last position
            last_token_probs = torch.softmax(last_token_logits, dim=-1)
            result["last_token_probs"] = last_token_probs
            result["last_token_entropy"] = -(last_token_probs * torch.log(last_token_probs + 1e-12)).sum()
            
            # Top predictions for next token
            top_k = torch.topk(last_token_probs, k=10)
            result["next_token_predictions"] = {
                "token_ids": top_k.indices,
                "tokens": [self.tokenizer.decode([tid]) for tid in top_k.indices],
                "probabilities": top_k.values
            }
        
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
