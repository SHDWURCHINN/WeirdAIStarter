"""
Pretrained model helpers for Weird AI.

Lesson 9 introduces students to using an already-trained language model for
text generation. Earlier lessons built many of the pieces from scratch. In this
lesson, students use a pretrained model so they can focus on the complete
inference pipeline:

    text prompt -> tokenizer -> token IDs -> model -> generated token IDs -> text

The functions in this file are intentionally scaffolded. Complete the TODO
sections, then run:

    python -m pytest tests/test_pretrained.py

The unit tests use small fake model and tokenizer objects, so they do not need
to download a real Hugging Face model.
"""

from __future__ import annotations

from typing import Any

import torch

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:  # pragma: no cover - handled by student setup or tests
    AutoModelForCausalLM = None
    AutoTokenizer = None

from weird_ai.reasoning import build_reasoning_prompt


DEFAULT_MODEL_NAME = "sshleifer/tiny-gpt2"


def get_device(preferred_device: str | None = None) -> torch.device:
    """Choose the device that should be used for model inference."""
    if preferred_device is not None:
        return torch.device(preferred_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def load_tokenizer(model_name: str = DEFAULT_MODEL_NAME) -> Any:
    """Load a pretrained tokenizer."""
    if AutoTokenizer is None:
        raise ImportError("transformers must be installed to load a tokenizer.")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(
    model_name: str = DEFAULT_MODEL_NAME,
    device: str | torch.device | None = None,
) -> Any:
    """Load a pretrained causal language model for text generation."""
    if AutoModelForCausalLM is None:
        raise ImportError("transformers must be installed to load a model.")
    resolved_device = get_device(str(device) if device is not None else None)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(resolved_device)
    model.eval()
    return model


def tokenize_prompt(
    tokenizer: Any,
    prompt: str,
    device: str | torch.device | None = None,
) -> dict[str, torch.Tensor]:
    """Convert a text prompt into model input tensors."""
    inputs = tokenizer(prompt, return_tensors="pt")
    if device is not None:
        resolved_device = get_device(str(device) if device is not None else None)
        inputs = {k: v.to(resolved_device) for k, v in inputs.items()}
    return inputs


def decode_generated_tokens(
    tokenizer: Any,
    generated_token_ids: torch.Tensor,
    skip_special_tokens: bool = True,
) -> str:
    """Decode generated token IDs into text."""
    if generated_token_ids.ndim > 1:
        token_ids = generated_token_ids[0]
    else:
        token_ids = generated_token_ids
    return tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)


def generate_text(
    prompt: str,
    model: Any | None = None,
    tokenizer: Any | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    max_new_tokens: int = 60,
    temperature: float = 0.8,
    top_k: int = 50,
    do_sample: bool = True,
    device: str | torch.device | None = None,
) -> str:
    """Generate text from a prompt using a pretrained causal language model."""
    resolved_device = get_device(str(device) if device is not None else None)
    if tokenizer is None:
        tokenizer = load_tokenizer(model_name)
    if model is None:
        model = load_model(model_name, device=resolved_device)
    
    inputs = tokenize_prompt(tokenizer, prompt, device=resolved_device)
    
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            do_sample=do_sample,
            pad_token_id=getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", None),
        )
    
    return decode_generated_tokens(tokenizer, output_ids)


def generate_with_reasoning(
    user_prompt: str,
    model: Any | None = None,
    tokenizer: Any | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    device: str | torch.device | None = None,
) -> str:
    """Generate text after wrapping the original prompt in a reasoning prompt."""
    reasoning_prompt = build_reasoning_prompt(user_prompt)
    return generate_text(
        prompt=reasoning_prompt,
        model=model,
        tokenizer=tokenizer,
        model_name=model_name,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        device=device,
    )