"""Measure intermediate LoRA scales on the four EXP-017 numeric-failure cells.

The deployment generator is not changed. Repair is not called. Scale 1.00 and
the adapter-off numeric result stay the saved EXP-017 draws. Blog voice at
scale 0.00 needs the base text, which that diagnostic did not keep, so the two
blog cells are drawn once more with the adapter disabled and the same seed.
"""
from __future__ import annotations

import json
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
from host_finetune.lora_scale import LoraScale
from host_finetune.voice_distance import style_distance

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments" / "EXP-20261004-017-relabel-grounding-repair" / "raw" / "relabel.jsonl"
CELLS = (
    ("gold_001", "blog"),
    ("gold_002", "blog"),
    ("gold_002", "linkedin"),
    ("gold_002", "talk"),
)
SCALES = (0.75, 0.50, 0.25)
SAVED_BASE_WORDS = {("gold_001", "blog"): 345, ("gold_002", "blog"): 384}
LEAK_MARKERS = (
    "do not mention the evidence",
    "these instructions",
    "allowed factual evidence",
    "unsupported numeric",
    "i removed the",
)


def load_cells(path: Path) -> dict[tuple[str, str], dict]:
    chosen = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (row.get("id"), row.get("medium"))
        if key not in CELLS:
            continue
        chosen[key] = row
    missing = [key for key in CELLS if key not in chosen]
    if missing:
        raise RuntimeError(f"missing scale cells: {missing}")
    return chosen


def _surfaces(items: list[dict] | None) -> list[str]:
    return [item.get("value") for item in (items or [])]


def record_from_saved(row: dict, scale: float) -> dict:
    numeric = row["numeric_grounding"]["first_attempt"]
    copy = row["rag_source_copying"]["first_attempt"]
    return _record(
        row,
        scale=scale,
        answer=row["raw_model_output"],
        numeric=numeric,
        copy=copy,
        stop_reason=row.get("stop_reason"),
        generated=False,
        adapter_enabled=scale != 0.0,
    )


def _record(row, *, scale, answer, numeric, copy, stop_reason, generated, adapter_enabled) -> dict:
    text = answer or ""
    folded = text.casefold()
    return {
        "id": row["id"],
        "medium": row["medium"],
        "topic": row["topic"],
        "scale": scale,
        "adapter_enabled": adapter_enabled,
        "generated_this_run": generated,
        "seed": row["seed"],
        "max_new_tokens": row["max_new_tokens"],
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "repetition_penalty": REPETITION_PENALTY,
        "do_sample": True,
        "prompt": row["prompt"],
        "hit_ids": [hit.get("id") for hit in row.get("retrieval_hits") or []],
        "accepted_claims": [claim.get("claim") for claim in row.get("verified_claims") or []],
        "answer": text,
        "word_count": len(text.split()),
        "stop_reason": stop_reason,
        "source_copy_overlap": copy.get("max_contiguous_words"),
        "source_copy_pass": copy.get("pass"),
        "quantities": _surfaces(numeric.get("values")),
        "unsupported_quantities": _surfaces(numeric.get("unsupported")),
        "numeric_pass": numeric.get("pass"),
        "instruction_leak": any(marker in folded for marker in LEAK_MARKERS),
        "repair": False,
    }


def _generate(model, tokenizer, row) -> dict:
    completion = _complete_once(
        model,
        tokenizer,
        row["prompt"],
        row["seed"],
        row["max_new_tokens"],
        row.get("retrieval_hits") or [],
        True,
    )
    return _attach_numeric(completion, row["topic"], row.get("verified_claims") or [], True)


def run(out_dir: Path, source: Path = SOURCE) -> list[dict]:
    cells = load_cells(source)
    records = [record_from_saved(cells[key], 1.0) for key in CELLS]
    import torch
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(ROOT / "host_finetune" / "output" / "lemkin_lora_relabel"),
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    control = LoraScale(model)
    print(
        f"scale probe layers={len(control.layers)} trained_factor={control.factors()[0]}",
        flush=True,
    )
    with control.contribution(0.5):
        if not all(abs(factor - 0.5) < 1e-6 for factor in control.factors()):
            raise RuntimeError("half scale did not apply to every LoRA layer")
    if not all(abs(factor - 1.0) < 1e-6 for factor in control.factors()):
        raise RuntimeError("scale did not return to 1.0 before generation")
    for factor in SCALES:
        for key in CELLS:
            row = cells[key]
            print(f"scale {factor} {key[0]} {key[1]}", flush=True)
            with control.contribution(factor):
                completion = _generate(model, tokenizer, row)
            if not all(abs(value - 1.0) < 1e-6 for value in control.factors()):
                raise RuntimeError("scale was not restored after a generation")
            records.append(
                _record(
                    row,
                    scale=factor,
                    answer=completion["answer"],
                    numeric=completion["numeric"],
                    copy=completion["report"],
                    stop_reason=completion.get("stop_reason"),
                    generated=True,
                    adapter_enabled=True,
                )
            )
    for key in (("gold_001", "blog"), ("gold_002", "blog")):
        row = cells[key]
        print(f"scale 0.00 adapter off {key[0]} blog", flush=True)
        completion = repair_under_disabled_adapter(model, lambda: _generate(model, tokenizer, row))
        saved_words = SAVED_BASE_WORDS[key]
        records.append(
            _record(
                row,
                scale=0.0,
                answer=completion["answer"],
                numeric=completion["numeric"],
                copy=completion["report"],
                stop_reason=completion.get("stop_reason"),
                generated=True,
                adapter_enabled=False,
            )
        )
        records[-1]["saved_diagnostic_words"] = saved_words
        records[-1]["reproduced_saved_word_count"] = records[-1]["word_count"] == saved_words
    del model
    torch.cuda.empty_cache()
    _add_blog_voice(records)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "raw").mkdir(exist_ok=True)
    (out_dir / "config").mkdir(exist_ok=True)
    (out_dir / "metrics").mkdir(exist_ok=True)
    (out_dir / "raw" / "scales.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
        encoding="utf-8",
    )
    (out_dir / "config" / "run_parameters.json").write_text(
        json.dumps(
            {
                "experiment_id": out_dir.name,
                "source_draws": "EXP-20261004-017-relabel-grounding-repair",
                "scales_generated": list(SCALES),
                "scale_1_source": "saved raw Relabel draw",
                "scale_0_blogs": "regenerated adapter off, same seed, for voice",
                "repair": False,
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
    return records


def _add_blog_voice(records: list[dict]) -> None:
    from host_finetune.compare_adapters import _evaluation_resources

    pack, _copy_index = _evaluation_resources()
    blog = pack["media"]["blog"]
    profile = blog["style_profile"]
    for row in records:
        row["blog_voice_role"] = blog["style_role"]
        if row["medium"] != "blog":
            row["blog_voice_distance"] = None
            continue
        distance = style_distance(row["answer"], profile)
        row["blog_voice_distance"] = None if distance is None else round(distance, 4)


def main() -> None:
    out = ROOT / "experiments" / "EXP-20261004-018-lora-scale"
    if out.exists():
        raise SystemExit(f"{out} already exists")
    run(out)


if __name__ == "__main__":
    main()
