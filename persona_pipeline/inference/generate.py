"""
Phase 9 — Content generation in Jason Lemkin's voice.

Given a topic and format type, uses PersonaRAG retrieval + the fine-tuned
LoRA model to generate B2B content.  Produces 3 candidates and returns the
one with the highest persona vocabulary overlap (draft ranking).

Run: python persona_pipeline/inference/generate.py
"""
from __future__ import annotations

import json
import os
import re
import sys
import warnings
from datetime import datetime
from pathlib import Path

REPO_ROOT = str(Path(__file__).resolve().parents[2])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from persona_pipeline.config.config import (
    LORA_OUTPUT,
    MODEL_NAME,
    PERSONA_PIPELINE_DIR,
    PERSONA_PROFILE_PATH,
)
from persona_pipeline.jobs.persona_rag import (
    load_persona_profile,
    persona_conditioned_retrieve,
)


OUTPUTS_DIR = PERSONA_PIPELINE_DIR.parent / "outputs"
FORMATS_DIR = PERSONA_PIPELINE_DIR / "config" / "formats"

NUM_CANDIDATES = 3


def _load_format_yaml(format_type: str) -> dict:
    path = FORMATS_DIR / f"{format_type}.yaml"
    if not path.exists():
        warnings.warn(f"Format YAML not found: {path}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_prompt(
    topic: str,
    format_type: str,
    persona_profile: dict,
    format_spec: dict,
    retrieved_chunks: list[dict],
) -> str:
    # a. System instruction
    style = persona_profile.get("style", {})
    patterns = persona_profile.get("rhetorical_patterns", [])
    pattern_str = "; ".join(patterns[:4])
    char_words = ", ".join(style.get("characteristic_words", [])[:15])

    system = (
        f"Write in the style of Jason Lemkin (SaaStr). "
        f"Use these characteristics: {pattern_str}. "
        f"Characteristic vocabulary: {char_words}."
    )

    # b. Format spec
    constraints = format_spec.get("constraints", {})
    tone_items = format_spec.get("tone", [])
    prohibited = format_spec.get("prohibited_patterns", [])
    structure = format_spec.get("structure", [])

    format_instr = (
        f"Format: {format_spec.get('format_name', format_type)}. "
        f"Word count: {constraints.get('min_words', 100)}-{constraints.get('max_words', 500)}. "
        f"Structure: {'; '.join(structure)}. "
        f"Tone: {'; '.join(tone_items)}. "
        f"Do NOT use these phrases: {', '.join(prohibited)}."
    )

    # c. Retrieved context
    context_parts = [r["text"][:500] for r in retrieved_chunks if r.get("text")]
    context_str = "\n---\n".join(context_parts) if context_parts else "(no context available)"

    # d. User request
    user_request = f"Now write a {format_type.replace('_', ' ')} about: {topic}"

    prompt = (
        f"### System:\n{system}\n\n"
        f"### Format:\n{format_instr}\n\n"
        f"### Context (relevant writing from Jason Lemkin):\n{context_str}\n\n"
        f"### Request:\n{user_request}\n\n"
        f"### Response:\n"
    )
    return prompt


def _persona_vocab_overlap(text: str, characteristic_words: list[str]) -> float:
    if not text or not characteristic_words:
        return 0.0
    words = set(re.findall(r"[a-z]{4,}", text.lower()))
    char_set = set(characteristic_words)
    return len(words & char_set) / max(len(char_set), 1)


def _compliance_check(text: str, format_spec: dict) -> list[str]:
    issues: list[str] = []
    constraints = format_spec.get("constraints", {})
    word_count = len(text.split())

    min_w = constraints.get("min_words", 0)
    max_w = constraints.get("max_words", 99999)
    if word_count < min_w:
        issues.append(f"Too short: {word_count} words (min {min_w})")
    if word_count > max_w:
        issues.append(f"Too long: {word_count} words (max {max_w})")

    for pattern in format_spec.get("prohibited_patterns", []):
        if pattern.lower() in text.lower():
            issues.append(f"Prohibited pattern found: '{pattern}'")

    return issues


def generate_content(
    topic: str,
    format_type: str = "linkedin_post",
    persona_profile_path: str | None = None,
    model_path: str | None = None,
) -> str:
    profile_path = persona_profile_path or PERSONA_PROFILE_PATH
    m_path = model_path or LORA_OUTPUT

    # 1. Load persona profile
    profile = load_persona_profile(profile_path)

    # 2. Load format YAML
    format_spec = _load_format_yaml(format_type)

    # 3. Retrieve context via PersonaRAG
    retrieved = persona_conditioned_retrieve(topic, profile, n_results=5)

    # 4. Build prompt
    prompt = _build_prompt(topic, format_type, profile, format_spec, retrieved)

    # 5. Load fine-tuned model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(m_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, m_path)
    model.eval()

    # 6. Generate candidates
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    char_words = profile.get("style", {}).get("characteristic_words", [])
    candidates: list[str] = []

    for _ in range(NUM_CANDIDATES):
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=500,
                temperature=0.8,
                do_sample=True,
                top_p=0.9,
            )
        generated = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        candidates.append(generated.strip())

    # 7. Draft ranking: pick candidate with highest persona vocab overlap
    scored = [
        (c, _persona_vocab_overlap(c, char_words))
        for c in candidates
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    best_text = scored[0][0]

    # 8. Compliance checks
    issues = _compliance_check(best_text, format_spec)
    if issues:
        for issue in issues:
            warnings.warn(f"Compliance issue: {issue}")

    # 9. Save output
    os.makedirs(str(OUTPUTS_DIR), exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUTS_DIR / f"{format_type}_{ts}.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# {topic}\n\n")
        f.write(best_text)
    print(f"Saved output -> {out_path}")

    return best_text


def main() -> None:
    topic = "Why every SaaS founder should hire two sales reps at the same time"
    fmt = "linkedin_post"
    print(f"Generating {fmt} about: {topic}\n")

    try:
        text = generate_content(topic, fmt)
        print("\n=== Generated Content ===")
        print(text)
    except Exception as e:
        warnings.warn(
            f"Generation failed (this is expected if the LoRA model hasn't been trained yet): {e}"
        )


if __name__ == "__main__":
    main()
