from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Iterable, Union

import inspect
import threading

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
)
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList


# -------------------------------
# 1) Safer dtype handling
# -------------------------------

def _pick_dtype(dtype: Optional[Union[str, torch.dtype]]) -> Union[torch.dtype, str]:
    """
    dtype = "auto" | "bfloat16"/"bf16" | "float16"/"fp16"/"half" | "float32"/"fp32"/"float" | None | torch.dtype

    Returns a torch.dtype or the string "auto".
    Raises ValueError for unknown strings (clearer than AttributeError).
    """
    if dtype is None or dtype == "auto":
        return "auto"
    if isinstance(dtype, torch.dtype):
        return dtype

    aliases = {
        "bf16": "bfloat16",
        "fp16": "float16",
        "half": "float16",
        "fp32": "float32",
        "float": "float32",
    }
    norm = aliases.get(dtype, dtype)
    valid = {"bfloat16", "float16", "float32"}
    if norm not in valid or not hasattr(torch, norm):
        raise ValueError(
            f"Unsupported dtype '{dtype}'. "
            f"Use one of: 'auto', 'bfloat16'/'bf16', 'float16'/'fp16'/'half', 'float32'/'fp32'/'float', "
            f"or an actual torch.dtype."
        )
    return getattr(torch, norm)


# -------------------------------
# 2) Tokenization helpers
# -------------------------------

def _apply_chat_template(tokenizer, messages: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Encode OpenAI-style messages using the tokenizer's chat template (Harmony-compatible).
    Falls back to a simple join if template is missing.
    NOTE: We DO NOT move tensors to a device here; keep CPU and let a later helper decide.
    """
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )

    # Fallback (plain text rendering)
    joined = []
    sys_chunks = [m["content"] for m in messages if m.get("role") == "system"]
    if sys_chunks:
        joined.append("SYSTEM: " + "\n".join(sys_chunks).strip())
    for m in messages:
        role = m.get("role", "user")
        if role == "system":
            continue
        role_up = "ASSISTANT" if role == "assistant" else "USER"
        joined.append(f"{role_up}: {m['content']}")
    joined.append("ASSISTANT:")
    text = "\n".join(joined)
    return tokenizer(text, return_tensors="pt")


def _infer_single_device(model) -> Optional[torch.device]:
    """Try to infer a single device for the whole model (None if sharded/offloaded)."""
    # Accelerate-sharded models expose hf_device_map; if present, assume multi-device/offload.
    if getattr(model, "hf_device_map", None):
        return None
    try:
        p = next(model.parameters())
        return p.device
    except StopIteration:
        return None
    except Exception:
        return None


def _prepare_inputs_for_model(inputs: Dict[str, Any], model) -> Dict[str, Any]:
    """
    Robust device placement:
      - If model appears single-device, move inputs to that device.
      - If sharded/offloaded, keep inputs on CPU and let HF handle dispatch.
    """
    device = _infer_single_device(model)
    if device is not None:
        return inputs.to(device)
    return inputs


# -------------------------------
# 3) Stop sequences support
# -------------------------------

class _StopOnSequences(StoppingCriteria):
    """
    Stop when any of the provided token-id sequences appears at the end of the generated sequence.
    """
    def __init__(self, stop_sequences: List[List[int]]):
        super().__init__()
        # Filter out empty sequences
        self.stop_sequences = [seq for seq in stop_sequences if len(seq) > 0]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if not self.stop_sequences:
            return False
        seq = input_ids[0].tolist()
        for stop in self.stop_sequences:
            L = len(stop)
            if L <= len(seq) and seq[-L:] == stop:
                return True
        return False


def _build_stopping_criteria(tokenizer, stop: Optional[List[str]]) -> Optional[StoppingCriteriaList]:
    if not stop:
        return None
    stop_ids: List[List[int]] = [tokenizer.encode(s, add_special_tokens=False) for s in stop]
    return StoppingCriteriaList([_StopOnSequences(stop_ids)])


# -------------------------------
# 4) Model wrapper
# -------------------------------

@dataclass
class GPTOSS:
    model: Any
    tokenizer: Any
    model_name_or_path: str

    # ---------- small utilities ----------

    def _ensure_padding_sane(self) -> None:
        """
        Ensure pad_token_id exists for decoder-only models (common: set pad=eos) and
        use left padding to avoid shifting issues in generation.
        """
        tok = self.tokenizer
        if tok.pad_token_id is None and tok.eos_token_id is not None:
            tok.pad_token = tok.eos_token
        # Left padding is recommended for decoder-only models
        if getattr(tok, "model_input_names", None) and "attention_mask" in tok.model_input_names:
            tok.padding_side = "left"

    # ---------- chat APIs ----------

    @torch.inference_mode()
    def chat(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 256,
        temperature: float = 0.2,
        top_p: float = 0.95,
        do_sample: Optional[bool] = None,
        stop: Optional[List[str]] = None,
        seed: Optional[int] = None,
        **gen_kwargs,
    ) -> str:
        """
        Simple chat; uses chat template if available.
        Adds 'stop' sequences and 'seed' for reproducibility.
        """
        self._ensure_padding_sane()

        if seed is not None:
            # For reproducibility when sampling
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        inputs = _apply_chat_template(self.tokenizer, messages)
        inputs = _prepare_inputs_for_model(inputs, self.model)

        stopping = _build_stopping_criteria(self.tokenizer, stop)
        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=(temperature > 0.0 if do_sample is None else do_sample),
            return_dict_in_generate=True,
            output_scores=False,
            stopping_criteria=stopping,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **gen_kwargs,
        )

        # Decode only newly generated tokens
        start = inputs["input_ids"].shape[-1]
        return self.tokenizer.decode(out.sequences[0][start:], skip_special_tokens=True)

    @torch.inference_mode()
    def stream_chat(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 256,
        temperature: float = 0.2,
        top_p: float = 0.95,
        do_sample: Optional[bool] = None,
        stop: Optional[List[str]] = None,
        **gen_kwargs,
    ) -> Iterable[str]:
        """
        Stream tokens as they are generated (yields text chunks).
        Useful for interactive UIs and debugging.
        """
        self._ensure_padding_sane()
        inputs = _apply_chat_template(self.tokenizer, messages)
        inputs = _prepare_inputs_for_model(inputs, self.model)

        stopping = _build_stopping_criteria(self.tokenizer, stop)
        streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True, skip_prompt=True)

        gen_args = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=(temperature > 0.0 if do_sample is None else do_sample),
            return_dict_in_generate=True,
            output_scores=False,
            stopping_criteria=stopping,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            streamer=streamer,
        )
        gen_args.update(gen_kwargs)

        thread = threading.Thread(target=self.model.generate, kwargs=gen_args)
        thread.start()
        for chunk in streamer:
            yield chunk
        thread.join()

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
            input_ids=input_ids.to(_infer_single_device(self.model) or input_ids.device),
            return_dict_in_generate=True,
            **gen_kwargs,
        )

    # ---------- inspection API ----------

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
        top_k: int = 10,
        compute_entropy: bool = False,
        stop: Optional[List[str]] = None,
        **forward_kwargs,
    ) -> Dict[str, Any]:
        """
        Forward pass with optional generation to expose model internals and response.

        New:
          - Device/offload safe inputs
          - Conditional 'output_router_logits' (won't crash non-MoE models)
          - JSON-safe intermediates with configurable 'top_k' and optional 'compute_entropy'
          - Optional 'stop' sequences
          - Sequence-level avg_logprob and perplexity (if intermediates requested)
        """
        self._ensure_padding_sane()

        inputs = _apply_chat_template(self.tokenizer, messages)
        inputs = _prepare_inputs_for_model(inputs, self.model)
        input_length = inputs["input_ids"].shape[-1]

        # Build forward kwargs that the model's forward actually supports
        def _filter_kwargs_for_forward(model, **kws):
            sig = inspect.signature(model.forward)
            allowed = set(sig.parameters.keys())
            return {k: v for k, v in kws.items() if k in allowed}

        if not generate:
            # Single forward pass (original behavior)
            call_kwargs = dict(
                use_cache=False,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
                return_dict=True,
            )
            if output_router_logits:
                call_kwargs["output_router_logits"] = True
            call_kwargs.update(forward_kwargs)
            call_kwargs = _filter_kwargs_for_forward(self.model, **call_kwargs)

            outputs = self.model(
                **inputs,
                **call_kwargs,
            )

            result = self._extract_outputs(outputs, inputs)
            result["input_tokens"] = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            return result

        # Generation with intermediates capture
        result: Dict[str, Any] = {
            "input_ids": inputs["input_ids"],
            "input_tokens": self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]),
            "generation_intermediates": [] if return_intermediates else None,
        }

        stopping = _build_stopping_criteria(self.tokenizer, stop)

        gen_outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0.0,
            return_dict_in_generate=True,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            output_scores=True,  # Per-token logits
            stopping_criteria=stopping,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
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
            result["scores"] = gen_outputs.scores  # List[T] per step

        # Token-level analysis
        if return_intermediates and hasattr(gen_outputs, "scores") and gen_outputs.scores:
            selected_logprobs: List[float] = []
            for i, token_scores in enumerate(gen_outputs.scores):
                token_id = int(generated_ids[i].item())
                logits = token_scores[0]                      # (vocab,)
                probs = torch.softmax(logits, dim=-1)         # (vocab,)
                prob = float(probs[token_id].item())
                logprob = float(torch.log(probs[token_id] + 1e-12).item())
                selected_logprobs.append(logprob)

                tk = min(top_k, probs.shape[-1])
                topk_vals, topk_ids = torch.topk(probs, k=tk)
                topk_vals = topk_vals.tolist()
                topk_ids = topk_ids.tolist()
                topk = [
                    {"token_id": int(tid),
                     "token": self.tokenizer.decode([tid]),
                     "probability": float(p)}
                    for tid, p in zip(topk_ids, topk_vals)
                ]

                intermediate = {
                    "step": i,
                    "token_id": token_id,
                    "token": self.tokenizer.decode([token_id]),
                    "probability": prob,
                    "logprob": logprob,
                    "top_k": topk,
                }
                if compute_entropy:
                    ent = float((-(probs * torch.log(probs + 1e-12))).sum().item())
                    intermediate["entropy"] = ent

                result["generation_intermediates"].append(intermediate)

            # Sequence-level aggregates
            if selected_logprobs:
                avg_logprob = float(sum(selected_logprobs) / len(selected_logprobs))
                # Perplexity over generated tokens
                result["avg_logprob"] = avg_logprob
                result["perplexity"] = float(torch.exp(torch.tensor(-avg_logprob)).item())

        return result

    def _extract_outputs(self, outputs, inputs) -> Dict[str, Any]:
        """Extract available outputs from model forward pass."""
        result: Dict[str, Any] = {}

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
                "probabilities": top_k.values,
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


# -------------------------------
# 5) Loader improvements
# -------------------------------

def load_gptoss(
    model_name_or_path: str = "openai/gpt-oss-20b",
    *,
    dtype: Union[str, torch.dtype, None] = "auto",
    device_map: Union[str, Dict[str, int], None] = "auto",
    attn_implementation: Optional[str] = None,
    trust_remote_code: bool = True,
    revision: Optional[str] = None,
    **model_kwargs,
) -> GPTOSS:
    """
    Loader tuned for GPT-OSS.

    Improvements:
      - Safer dtype normalization & validation
      - Auto pad_token fix and left padding for decoder-only models
      - Optional 'revision' pin for reproducibility
      - model.eval() for inference mode
    """
    tok = AutoTokenizer.from_pretrained(
        model_name_or_path,
        revision=revision,
        trust_remote_code=trust_remote_code,
    )
    # Safety: set pad_token if missing; left padding for decoder-only
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    kwargs = dict(
        torch_dtype=_pick_dtype(dtype),
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        revision=revision,
        low_cpu_mem_usage=True,
    )
    if attn_implementation:
        # e.g., "flash_attention_2" or "sdpa"; value must be valid for the model
        kwargs["attn_implementation"] = attn_implementation

    kwargs.update(model_kwargs)

    mdl = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
    mdl.eval()
    return GPTOSS(model=mdl, tokenizer=tok, model_name_or_path=model_name_or_path)
