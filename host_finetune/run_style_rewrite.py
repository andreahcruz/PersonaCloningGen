"""Eight-cell sanity for Base grounded draft plus one Relabel style rewrite.

Evidence is the frozen EXP-017 retrieval and accepted claims. The deployment
generator is not changed, and this run does not repair or retry.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from host_finetune.compare_adapters import (
    REPETITION_PENALTY,
    TEMPERATURE,
    TOP_P,
    _attach_numeric,
    _complete_once,
    git_state,
)
from host_finetune.grounding_repair import repair_under_disabled_adapter
from host_finetune.style_rewrite import (
    content_changes,
    grounded_base_prompt,
    select_deployment,
    stage_failure,
    style_rewrite_prompt,
)
from host_finetune.voice_distance import style_distance

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments" / "EXP-20261004-017-relabel-grounding-repair" / "raw" / "relabel.jsonl"
ORDER = (
    ("gold_001", "blog"),
    ("gold_001", "linkedin"),
    ("gold_001", "x"),
    ("gold_001", "talk"),
    ("gold_002", "blog"),
    ("gold_002", "linkedin"),
    ("gold_002", "x"),
    ("gold_002", "talk"),
)


def load_cells(path: Path) -> dict[tuple[str, str], dict]:
    chosen = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (row.get("id"), row.get("medium"))
        if key in ORDER:
            chosen[key] = row
    missing = [key for key in ORDER if key not in chosen]
    if missing:
        raise RuntimeError(f"missing cells: {missing}")
    return chosen


def _surfaces(items) -> list[str]:
    return [item.get("value") for item in (items or [])]


def _stage_view(answer: str, numeric: dict, copy: dict, stop_reason: str | None, seconds: float) -> dict:
    return {
        "answer": answer,
        "word_count": len((answer or "").split()),
        "stop_reason": stop_reason,
        "seconds": round(seconds, 3),
        "quantities": _surfaces(numeric.get("values")),
        "unsupported_quantities": _surfaces(numeric.get("unsupported")),
        "source_copy_overlap": copy.get("max_contiguous_words"),
        "source_copy_pass": copy.get("pass"),
        "numeric_pass": numeric.get("pass"),
        "failure": stage_failure(answer, numeric, copy),
    }


def _generate(model, tokenizer, instruction: str, row: dict) -> tuple[dict, float]:
    started = time.perf_counter()
    completion = _complete_once(
        model,
        tokenizer,
        instruction,
        row["seed"],
        row["max_new_tokens"],
        row.get("retrieval_hits") or [],
        True,
    )
    elapsed = time.perf_counter() - started
    attached = _attach_numeric(completion, row["topic"], row.get("verified_claims") or [], True)
    return attached, elapsed


def run(out_dir: Path, source: Path = SOURCE) -> list[dict]:
    cells = load_cells(source)
    import torch
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(ROOT / "host_finetune" / "output" / "lemkin_lora_relabel"),
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    _require_trained_scale(model)
    records = []
    for key in ORDER:
        row = cells[key]
        claims = row.get("verified_claims") or []
        hits = row.get("retrieval_hits") or []
        base_prompt = grounded_base_prompt(row["medium"], row["topic"], claims)
        _refuse_leaked_prompt(base_prompt, claims, hits)
        print(f"base {key[0]} {key[1]}", flush=True)
        base, base_seconds = repair_under_disabled_adapter(
            model, lambda: _generate(model, tokenizer, base_prompt, row)
        )
        base_view = _stage_view(
            base["answer"], base["numeric"], base["report"], base.get("stop_reason"), base_seconds
        )
        rewrite_view = None
        changes = None
        style_prompt = None
        if base_view["failure"]:
            print(f"  base stopped {key[0]} {key[1]} {base_view['failure']}", flush=True)
            selected = select_deployment(base["answer"], base_view["failure"], None, None)
        else:
            style_prompt = style_rewrite_prompt(row["medium"], row["topic"], base["answer"], claims)
            _refuse_leaked_prompt(style_prompt, claims, hits)
            print(f"style {key[0]} {key[1]}", flush=True)
            styled, style_seconds = _generate(model, tokenizer, style_prompt, row)
            rewrite_view = _stage_view(
                styled["answer"],
                styled["numeric"],
                styled["report"],
                styled.get("stop_reason"),
                style_seconds,
            )
            changes = content_changes(base["answer"], styled["answer"])
            selected = select_deployment(
                base["answer"], None, styled["answer"], rewrite_view["failure"]
            )
        records.append({
            "id": row["id"],
            "medium": row["medium"],
            "topic": row["topic"],
            "system": "Base + Factual RAG + Relabel Style Rewrite",
            "seed": row["seed"],
            "max_new_tokens": row["max_new_tokens"],
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "repetition_penalty": REPETITION_PENALTY,
            "do_sample": True,
            "evidence_source": "EXP-20261004-017-relabel-grounding-repair",
            "hit_ids": [hit.get("id") for hit in hits],
            "retrieval_hits": hits,
            "verified_claims": claims,
            "extraction_seconds": row.get("extraction_seconds"),
            "base_prompt": base_prompt,
            "style_prompt": style_prompt,
            "grounded_base_draft": base_view,
            "relabel_style_rewrite": rewrite_view,
            "content_changes": changes,
            "deployment_output": selected["deployment_output"],
            "deployment_stage": selected["deployment_stage"],
            "style_rewrite_accepted": selected["style_rewrite_accepted"],
            "fallback_reason": selected["fallback_reason"],
            "stopped_reason": selected["stopped_reason"],
            "repair": False,
            "retry": False,
        })
        print(
            f"  {key[0]} {key[1]} base_pass={base_view['failure'] is None} "
            f"rewrite_accepted={selected['style_rewrite_accepted']} "
            f"stage={selected['deployment_stage']}",
            flush=True,
        )
    del model
    torch.cuda.empty_cache()
    _add_blog_voice(records)
    _write(out_dir, records)
    return records


def _require_trained_scale(model) -> None:
    for module in model.modules():
        scaling = getattr(module, "scaling", None)
        alpha = getattr(module, "lora_alpha", None)
        rank = getattr(module, "r", None)
        if not isinstance(scaling, dict) or not scaling or not isinstance(alpha, dict):
            continue
        for name, value in scaling.items():
            expected = float(alpha[name]) / float(rank[name])
            if abs(float(value) - expected) > 1e-5:
                raise RuntimeError(f"LoRA scale is {value}, not the trained {expected}")
        return
    raise RuntimeError("no LoRA scale found")


def _refuse_leaked_prompt(prompt: str, claims: list[dict], hits: list[dict]) -> None:
    for hit in hits:
        text = (hit.get("text") or "").strip()
        if len(text) >= 40 and text in prompt:
            raise RuntimeError("raw retrieved passage entered a generation prompt")
    for claim in claims:
        span = (claim.get("support_span") or "").strip()
        claim_text = claim.get("claim") or ""
        if span and span not in claim_text and span in prompt:
            raise RuntimeError("support span entered a generation prompt")


def _add_blog_voice(records: list[dict]) -> None:
    from host_finetune.compare_adapters import _evaluation_resources

    pack, _copy_index = _evaluation_resources()
    blog = pack["media"]["blog"]
    profile = blog["style_profile"]
    for row in records:
        row["blog_voice_role"] = blog["style_role"]
        if row["medium"] != "blog":
            row["base_blog_voice"] = None
            row["rewrite_blog_voice"] = None
            continue
        row["base_blog_voice"] = _distance(row["grounded_base_draft"]["answer"], profile)
        rewrite = row.get("relabel_style_rewrite")
        row["rewrite_blog_voice"] = None if rewrite is None else _distance(rewrite["answer"], profile)


def _distance(text: str, profile: dict):
    distance = style_distance(text, profile)
    return None if distance is None else round(distance, 4)


def _write(out_dir: Path, records: list[dict]) -> None:
    out_dir.mkdir(parents=True)
    (out_dir / "raw").mkdir()
    (out_dir / "config").mkdir()
    (out_dir / "metrics").mkdir()
    (out_dir / "raw" / "style_rewrite.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
        encoding="utf-8",
    )
    (out_dir / "config" / "run_parameters.json").write_text(
        json.dumps(
            {
                "experiment_id": out_dir.name,
                "system": "Base + Factual RAG + Relabel Style Rewrite",
                "not_equivalent_to": "Relabel + Factual RAG",
                "evidence_source": "EXP-20261004-017-relabel-grounding-repair",
                "base_adapter_enabled": False,
                "rewrite_adapter_scale": 1.0,
                "retry": False,
                "repair": False,
                "fallback": "grounded_base_draft",
                "git": git_state(),
                "generation": {
                    "temperature": TEMPERATURE,
                    "top_p": TOP_P,
                    "repetition_penalty": REPETITION_PENALTY,
                    "do_sample": True,
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    out = ROOT / "experiments" / "EXP-20261005-001-base-style-rewrite"
    if out.exists():
        raise SystemExit(f"{out} already exists")
    run(out)


if __name__ == "__main__":
    main()
