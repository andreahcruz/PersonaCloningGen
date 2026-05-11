"""Shared Llama 3.x Instruct chat strings for training and Ollama inference.

Training uses ``tokenizer.apply_chat_template``; Ollama must use a Modelfile
``TEMPLATE`` that matches the same control tokens (see ``host_finetune/Modelfile``).
"""
from __future__ import annotations

import os
from typing import Any

# Default system block embedded in Meta's Llama 3.1 Instruct template (must stay in
# sync with the tokenizer’s built-in chat template when no explicit system message).
DEFAULT_LLAMA31_SYSTEM_BLOCK = (
    "Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024"
)


def user_message_from_alpaca_row(instruction: str, input_text: str) -> str:
    inst = (instruction or "").strip()
    inp = (input_text or "").strip()
    if not inp:
        return inst
    return f"{inst}\n\n{inp}"


def training_text_from_row(
    tokenizer: Any,
    instruction: str,
    input_text: str,
    output: str,
) -> str:
    """Full multi-turn string as used in SFT (no generation prompt)."""
    messages = [
        {"role": "user", "content": user_message_from_alpaca_row(instruction, input_text)},
        {"role": "assistant", "content": (output or "").strip()},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def unsloth_response_markers(tokenizer: Any) -> tuple[str, str]:
    """(instruction_part, response_part) for ``unsloth.chat_templates.train_on_responses_only``."""
    mark_u, mark_a = "\x7fUSER\x7f", "\x7fASST\x7f"
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": mark_u}, {"role": "assistant", "content": mark_a}],
        tokenize=False,
        add_generation_prompt=False,
    )
    iu, ia = rendered.index(mark_u), rendered.index(mark_a)
    eot = "<|eot_id|>"

    def _after_last_eot_before(pos: int) -> int:
        j = rendered.rfind(eot, 0, pos)
        if j < 0:
            raise ValueError(
                "Could not parse chat template: missing <|eot_id|> before a marker."
            )
        return j + len(eot)

    instruction_part = rendered[_after_last_eot_before(iu) : iu]
    response_part = rendered[_after_last_eot_before(ia) : ia]
    return instruction_part, response_part


def inference_prompt_from_user_text(tokenizer: Any, user_text: str) -> str:
    """Same prefix Ollama should see before the assistant generates (generation prompt)."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text.strip()}],
        tokenize=False,
        add_generation_prompt=True,
    )


def load_tokenizer_for_prompt_build() -> Any:
    """Light tokenizer load for smoke tests (no GPU weights)."""
    from transformers import AutoTokenizer

    name = os.environ.get(
        "HF_MODEL_NAME", "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
    )
    return AutoTokenizer.from_pretrained(name)
