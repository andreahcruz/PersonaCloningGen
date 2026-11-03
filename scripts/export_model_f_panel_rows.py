"""Export the official F-REL panel without editing the experiment directory.

Writes a full private package under private_handoff/ and a public package
under results/model-f-panel-rows-v1/. The public package omits draft text.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "EXP-20261102-001-model-f-unified-panel"
AUDIT = ROOT / "audits" / "AUDIT-20261102-001-model-f-frel-prejudge"
PRIVATE = ROOT / "private_handoff" / "model-f-panel-rows-v1"
PUBLIC = ROOT / "results" / "model-f-panel-rows-v1"
ZIP_PATH = ROOT / "private_handoff" / "model-f-frel-120-evaluation.zip"
MEDIA = ("blog", "linkedin", "x", "talk")
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
SECRET_RE = re.compile(
    r"(AKIA[0-9A-Z]{16}|sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|BEGIN (RSA |OPENSSH |PRIVATE) KEY)"
)
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

JUDGE_FIELDS = [
    "question_id",
    "medium",
    "eligible_for_ranking",
    "completion_pass",
    "mid_sentence_stop",
    "sentence_final",
    "format_pass",
    "format_word_count",
    "copying_pass",
    "copying_longest_shared_word_span",
    "grounding_pass",
    "grounding_claim_support_rate",
    "grounding_contradiction_rate",
    "grounding_supported",
    "grounding_unsupported",
    "grounding_contradicted",
    "grounding_not_verifiable",
    "grounding_not_applicable",
    "writing_quality",
    "writing_coherence",
    "writing_fluency",
    "writing_logical_progression",
    "writing_relevance",
    "writing_specificity",
    "writing_usefulness",
    "task_adherence",
    "task_brief_alignment",
    "style_distance",
    "stance_consistency",
    "framing_distance",
    "persona_utility",
    "lemkin_evidence_support_rate",
    "lemkin_evidence_contradiction_rate",
    "bleu",
    "rouge_l",
    "latency_seconds",
]

FLAG_FIELDS = [
    "question_id",
    "medium",
    "unsupported_numeric_count",
    "numeric_pass",
    "numeric_applicable",
    "numeric_value_count",
    "mid_sentence_stop",
    "at_max_tokens",
    "output_tokens",
    "max_new_tokens",
    "source_copy_pass",
    "source_copy_span_words",
    "source_copy_threshold",
    "training_copy_pass",
    "training_copy_span_words",
    "repeats_topic",
    "embeds_topic",
    "faithfulness_pass",
    "prompt_contract_short",
    "instruction_or_marker_leak",
    "citation_marker_leak",
    "format_pass",
    "completion_pass",
    "lexical_grounding_pass",
    "role_unsupported_count",
    "speaker_applicable",
    "speaker_pass",
    "accepted_claims",
    "accepted_attempt",
    "evidence_status",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_generations() -> list[dict]:
    paths = sorted((EXP / "raw" / "generations").glob("gold_*__F-REL__*.json"))
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def key_of(row: dict, id_field: str) -> tuple[str, str]:
    return (str(row[id_field]), str(row["medium"]))


def expect_panel(label: str, keys: list[tuple[str, str]]) -> None:
    expected = {(f"gold_{index:03d}", medium) for index in range(1, 31) for medium in MEDIA}
    if len(keys) != 120:
        raise SystemExit(f"{label} count {len(keys)} != 120")
    if len(set(keys)) != 120:
        raise SystemExit(f"{label} has duplicate cells")
    if set(keys) != expected:
        missing = sorted(expected - set(keys))
        extra = sorted(set(keys) - expected)
        raise SystemExit(f"{label} key mismatch missing={missing[:5]} extra={extra[:5]}")


def cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: cell(row.get(field)) for field in fields})


def judge_row(row: dict) -> dict:
    gates = row.get("gates") or {}
    completion = gates.get("completion") or {}
    copying = gates.get("copying_memorization") or {}
    formatting = gates.get("format") or {}
    gate_grounding = gates.get("grounding") or {}
    grounding = row.get("grounding") or {}
    counts = grounding.get("counts") or {}
    writing = row.get("writing_quality") or {}
    dimensions = writing.get("dimensions") or {}
    task = row.get("task_adherence") or {}
    style = row.get("persona_style") or {}
    perspective = row.get("perspective_fidelity") or {}
    lemkin = row.get("lemkin_evidence_support") or {}
    diagnostics = row.get("diagnostics") or {}
    reference = diagnostics.get("reference_similarity") or {}
    return {
        "question_id": row.get("id"),
        "medium": row.get("medium"),
        "eligible_for_ranking": row.get("eligible_for_ranking"),
        "completion_pass": completion.get("pass"),
        "mid_sentence_stop": completion.get("mid_sentence_stop"),
        "sentence_final": completion.get("sentence_final"),
        "format_pass": formatting.get("pass"),
        "format_word_count": formatting.get("word_count"),
        "copying_pass": copying.get("pass"),
        "copying_longest_shared_word_span": copying.get("longest_shared_word_span"),
        "grounding_pass": gate_grounding.get("pass"),
        "grounding_claim_support_rate": grounding.get("claim_support_rate"),
        "grounding_contradiction_rate": grounding.get("contradiction_rate"),
        "grounding_supported": counts.get("supported"),
        "grounding_unsupported": counts.get("unsupported"),
        "grounding_contradicted": counts.get("contradicted"),
        "grounding_not_verifiable": counts.get("not_verifiable"),
        "grounding_not_applicable": counts.get("not_applicable"),
        "writing_quality": writing.get("mean"),
        "writing_coherence": dimensions.get("coherence"),
        "writing_fluency": dimensions.get("fluency"),
        "writing_logical_progression": dimensions.get("logical_progression"),
        "writing_relevance": dimensions.get("relevance"),
        "writing_specificity": dimensions.get("specificity"),
        "writing_usefulness": dimensions.get("usefulness"),
        "task_adherence": task.get("score"),
        "task_brief_alignment": task.get("brief_alignment"),
        "style_distance": style.get("distance"),
        "stance_consistency": perspective.get("stance_consistency_rate"),
        "framing_distance": perspective.get("framing_distance"),
        "persona_utility": row.get("persona_utility"),
        "lemkin_evidence_support_rate": lemkin.get("claim_support_rate"),
        "lemkin_evidence_contradiction_rate": lemkin.get("contradiction_rate"),
        "bleu": reference.get("bleu"),
        "rouge_l": reference.get("rouge_l"),
        "latency_seconds": diagnostics.get("latency_seconds"),
    }


def flag_row(row: dict) -> dict:
    completion = row.get("completion") or {}
    training = row.get("training_copy_gate") or {}
    lexical = row.get("lexical_grounding") or {}
    speaker = row.get("speaker") or {}
    return {
        "question_id": row.get("question_id"),
        "medium": row.get("medium"),
        "unsupported_numeric_count": row.get("unsupported_numeric_count"),
        "numeric_pass": row.get("numeric_pass"),
        "numeric_applicable": row.get("numeric_applicable"),
        "numeric_value_count": row.get("numeric_value_count"),
        "mid_sentence_stop": completion.get("mid_sentence_stop"),
        "at_max_tokens": row.get("at_max_tokens"),
        "output_tokens": row.get("output_tokens"),
        "max_new_tokens": row.get("max_new_tokens"),
        "source_copy_pass": row.get("copy_pass"),
        "source_copy_span_words": row.get("copy_span"),
        "source_copy_threshold": row.get("copy_threshold"),
        "training_copy_pass": training.get("pass"),
        "training_copy_span_words": training.get("longest_shared_word_span"),
        "repeats_topic": row.get("repeats_topic"),
        "embeds_topic": row.get("embeds_topic"),
        "faithfulness_pass": row.get("faithfulness_pass"),
        "prompt_contract_short": row.get("prompt_contract_short"),
        "instruction_or_marker_leak": row.get("instruction_or_marker_leak"),
        "citation_marker_leak": row.get("citation_marker_leak"),
        "format_pass": (row.get("format") or {}).get("pass"),
        "completion_pass": completion.get("pass"),
        "lexical_grounding_pass": lexical.get("pass"),
        "role_unsupported_count": row.get("role_unsupported_count"),
        "speaker_applicable": speaker.get("applicable"),
        "speaker_pass": speaker.get("pass"),
        "accepted_claims": row.get("accepted_claims_file"),
        "accepted_attempt": row.get("accepted_attempt"),
        "evidence_status": row.get("evidence_status"),
    }


def generation_row(record: dict) -> dict:
    config = record.get("generation_config") or {}
    decoding = config.get("decoding") or {}
    tokens = record.get("tokens") or {}
    output_tokens = tokens.get("output")
    return {
        "question_id": record.get("question_id"),
        "medium": record.get("medium"),
        "response": record.get("answer"),
        "model_id": "F-REL",
        "base_model": "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        "adapter": "host_finetune/output/lemkin_lora_relabel",
        "prompt_family": config.get("prompt_family"),
        "quantization": config.get("quantization"),
        "max_seq_length": config.get("max_seq_length"),
        "seed": config.get("seed"),
        "temperature": decoding.get("temperature"),
        "top_p": decoding.get("top_p"),
        "repetition_penalty": decoding.get("repetition_penalty"),
        "max_new_tokens": config.get("max_new_tokens"),
        "output_tokens": output_tokens,
    }


def safety_review(generations: list[dict], flags: list[dict]) -> dict:
    url_cells = []
    email_cells = []
    secret_cells = []
    path_cells = []
    marker_cells = []
    for record in generations:
        answer = str(record.get("answer") or "")
        label = f"{record.get('question_id')}:{record.get('medium')}"
        if URL_RE.search(answer):
            url_cells.append(label)
        if EMAIL_RE.search(answer):
            email_cells.append(label)
        if SECRET_RE.search(answer):
            secret_cells.append(label)
        if "C:\\Users" in answer or "C:/Users" in answer:
            path_cells.append(label)
        if "[E1]" in answer or "FACTUALITY AND ATTRIBUTION" in answer:
            marker_cells.append(label)
    copy_fail = [
        f"{row['question_id']}:{row['medium']}"
        for row in flags
        if row.get("source_copy_pass") is False
    ]
    copy_span_8 = [
        f"{row['question_id']}:{row['medium']}"
        for row in flags
        if isinstance(row.get("source_copy_span_words"), int) and row["source_copy_span_words"] >= 8
    ]
    return {
        "drafts_reviewed": len(generations),
        "prompts_included_in_export": False,
        "retrieved_passages_included_in_export": False,
        "source_copy_fail_12_word_cells": copy_fail,
        "source_copy_span_at_least_8_cells": copy_span_8,
        "cells_with_urls": url_cells,
        "cells_with_email_addresses": email_cells,
        "cells_with_secret_patterns": secret_cells,
        "cells_with_local_paths": path_cells,
        "cells_with_prompt_markers": marker_cells,
        "public_generation_text": False,
        "reason": (
            "The GitHub repository is public. Seven saved drafts fail the existing "
            "12-word source-copy check, additional drafts share 8 or more words with a "
            "retrieved passage, and at least one draft contains a URL. Publication "
            "permission for these source-derived responses was not established. "
            "The private export keeps every response unchanged."
        ),
    }


def readme(private: bool) -> str:
    location = "private export" if private else "public package"
    text_note = (
        "This directory contains all 120 responses in `generations.jsonl`."
        if private
        else (
            "This public directory does not contain `generations.jsonl`. "
            "The 120 responses are in the private archive "
            "`model-f-frel-120-evaluation.zip`, unchanged. "
            "Public publication was withheld because source-copy flags, a URL inside a draft, "
            "and the lack of a publication grant make the response text unsafe for this public repository."
        )
    )
    return f"""# F-REL 120-cell row export ({location})

{text_note}

The official experiment directory was not modified. Scores and flags are copied from saved files. This export does not rescore them.

## Files

| File | Rows | Contents |
|---|---:|---|
| `generations.jsonl` | 120 | Private only. Question, medium, response text, model identifiers, recorded output token count |
| `judge_scores.csv` | 120 | Qwen 2.5 14B `unified_eval.v1` row from `unified_eval/rows.jsonl` |
| `deterministic_flags.csv` | 120 | Pre-judge audit cell from `audits/AUDIT-20261102-001-model-f-frel-prejudge/cells.jsonl` |
| `manifest.json` | — | Identities and SHA-256 hashes |

## Join

The key is `question_id` + `medium`. Question IDs are `gold_001` through `gold_030`. Media are `blog`, `linkedin`, `x`, and `talk`. Each file has one row per pair and no duplicate pairs.

## Field notes

Empty CSV cells are missing values. They are not zeros.

`output_tokens` is `tokens.output` on the generation record, with `tokens.source` recorded there as `caller_tokenizer`.

`writing_quality` is the judge mean. Dimension columns are the stored subscores. `style_distance` is empty when the saved row has no distance, which is the case for talk. `persona_utility` is empty on every Model F row.

`grounding_pass` and the grounding rates are the judge's claim-versus-saved-retrieval-hit score. `unsupported_numeric_count` is the deterministic quantity check. Those two columns do not measure the same thing.

`source_copy_pass` fails when the audit recorded a retrieved-passage overlap of at least 12 words. `training_copy_pass` is the separate training-span gate. A true training-copy pass can sit beside a false source-copy pass.

`mid_sentence_stop` comes from the audit completion object. `at_max_tokens` is the audit flag that output tokens equal the medium cap. The generation record does not store a stop reason.

## Evaluation limits

The judge is blinded `qwen2.5:14b` at temperature 0. It is not the Llama generator. Eligible writing quality and claim-support can look strong while the deterministic number check fails. On this panel the audit recorded 52 cells with an unsupported number. The judge grounding gate is a different test: it scores extracted claims against saved retrieval hits, and a draft can introduce quantities that never become those claims.

Stance is diagnostic. Persona utility was not computed. Style distance is not an authorship test.

The generation manifest still says `generated_not_judged`. The judge file is `unified_eval/rows.jsonl`.
"""


def main() -> None:
    generations = load_generations()
    judges = load_jsonl(EXP / "unified_eval" / "rows.jsonl")
    audits = load_jsonl(AUDIT / "cells.jsonl")
    gen_keys = [key_of(row, "question_id") for row in generations]
    judge_keys = [key_of(row, "id") for row in judges]
    audit_keys = [key_of(row, "question_id") for row in audits]
    expect_panel("generations", gen_keys)
    expect_panel("judge", judge_keys)
    expect_panel("audit", audit_keys)
    if set(gen_keys) != set(judge_keys) or set(gen_keys) != set(audit_keys):
        raise SystemExit("generation, judge, and audit keys differ")

    order = [(f"gold_{index:03d}", medium) for index in range(1, 31) for medium in MEDIA]
    by_gen = {key_of(row, "question_id"): row for row in generations}
    by_judge = {key_of(row, "id"): row for row in judges}
    by_audit = {key_of(row, "question_id"): row for row in audits}
    gen_rows = [generation_row(by_gen[key]) for key in order]
    judge_rows = [judge_row(by_judge[key]) for key in order]
    flag_rows = [flag_row(by_audit[key]) for key in order]
    if any(row["response"] in (None, "") for row in gen_rows):
        raise SystemExit("a generation response is empty")
    if any(row["output_tokens"] is None for row in gen_rows):
        raise SystemExit("a generation is missing tokens.output")

    recorded_sha = json.loads((EXP / "unified_eval" / "summary.json").read_text(encoding="utf-8"))["generation_sha256"]
    trace_sha = sha256_file(EXP / "raw" / "judge_inputs.jsonl")
    if trace_sha != recorded_sha:
        raise SystemExit(f"judge_inputs hash {trace_sha} != recorded {recorded_sha}")
    metrics_copy = sha256_file(EXP / "metrics" / "rows.jsonl")
    official_rows = sha256_file(EXP / "unified_eval" / "rows.jsonl")

    review = safety_review([by_gen[key] for key in order], flag_rows)
    if review["cells_with_secret_patterns"] or review["cells_with_local_paths"]:
        raise SystemExit(f"unexpected sensitive draft content: {review}")

    PRIVATE.mkdir(parents=True, exist_ok=True)
    PUBLIC.mkdir(parents=True, exist_ok=True)
    gen_text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in gen_rows)
    (PRIVATE / "generations.jsonl").write_text(gen_text, encoding="utf-8")
    write_csv(PRIVATE / "judge_scores.csv", JUDGE_FIELDS, judge_rows)
    write_csv(PRIVATE / "deterministic_flags.csv", FLAG_FIELDS, flag_rows)
    write_csv(PUBLIC / "judge_scores.csv", JUDGE_FIELDS, judge_rows)
    write_csv(PUBLIC / "deterministic_flags.csv", FLAG_FIELDS, flag_rows)
    (PRIVATE / "README.md").write_text(readme(private=True), encoding="utf-8")
    (PUBLIC / "README.md").write_text(readme(private=False), encoding="utf-8")
    (PRIVATE / "safety_review.json").write_text(
        json.dumps(review, indent=2) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "experiment_id": "EXP-20261102-001-model-f-unified-panel",
        "profile": "F-REL",
        "generation_sha256": recorded_sha,
        "generation_sha256_meaning": "SHA-256 of experiments/EXP-20261102-001-model-f-unified-panel/raw/judge_inputs.jsonl, recomputed and matched. That trace file contains prompts and is not in either export.",
        "gold_sha256": "d7c4f997899e28f28eb3ea9aa3417857da6f0666acbf72fd8ac31422034e520e",
        "judge_model": "qwen2.5:14b",
        "judge_protocol": "unified_eval.v1",
        "judge_temperature": 0,
        "judge_blinded": True,
        "rows": {
            "generations": 120,
            "judge_scores": 120,
            "deterministic_flags": 120,
        },
        "authoritative_sources": {
            "generations": "experiments/EXP-20261102-001-model-f-unified-panel/raw/generations/gold_*__F-REL__*.json",
            "judge_rows": "experiments/EXP-20261102-001-model-f-unified-panel/unified_eval/rows.jsonl",
            "audit_cells": "audits/AUDIT-20261102-001-model-f-frel-prejudge/cells.jsonl",
        },
        "metrics_rows_jsonl_sha256": metrics_copy,
        "unified_eval_rows_jsonl_sha256": official_rows,
        "metrics_rows_match_unified_eval_rows": metrics_copy == official_rows,
        "public_omits_generation_text": True,
        "safety": {
            "source_copy_fail_cells": len(review["source_copy_fail_12_word_cells"]),
            "source_copy_span_at_least_8_cells": len(review["source_copy_span_at_least_8_cells"]),
            "cells_with_urls": len(review["cells_with_urls"]),
            "cells_with_prompt_markers": len(review["cells_with_prompt_markers"]),
            "cells_with_email_addresses": len(review["cells_with_email_addresses"]),
            "cells_with_secret_patterns": len(review["cells_with_secret_patterns"]),
        },
    }
    private_files = {
        "generations.jsonl": PRIVATE / "generations.jsonl",
        "judge_scores.csv": PRIVATE / "judge_scores.csv",
        "deterministic_flags.csv": PRIVATE / "deterministic_flags.csv",
        "README.md": PRIVATE / "README.md",
        "safety_review.json": PRIVATE / "safety_review.json",
    }
    manifest["export_sha256"] = {name: sha256_file(path) for name, path in private_files.items()}
    (PRIVATE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (PUBLIC / "manifest.json").write_text((PRIVATE / "manifest.json").read_text(encoding="utf-8"), encoding="utf-8")

    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in (
            "generations.jsonl",
            "judge_scores.csv",
            "deterministic_flags.csv",
            "README.md",
            "safety_review.json",
            "manifest.json",
        ):
            archive.write(PRIVATE / name, arcname=f"model-f-panel-rows-v1/{name}")
        archive.write(
            EXP / "unified_eval" / "rows.jsonl",
            arcname="model-f-panel-rows-v1/source/unified_eval_rows.jsonl",
        )
        archive.write(AUDIT / "cells.jsonl", arcname="model-f-panel-rows-v1/source/audit_cells.jsonl")
    zip_sha = sha256_file(ZIP_PATH)
    second = sha256_file(ZIP_PATH)
    if zip_sha != second:
        raise SystemExit("zip hash changed between reads")
    with zipfile.ZipFile(ZIP_PATH) as archive:
        names = archive.namelist()
        bad = archive.testzip()
    if bad:
        raise SystemExit(f"zip corrupt at {bad}")
    receipt = {
        "zip": ZIP_PATH.name,
        "bytes": ZIP_PATH.stat().st_size,
        "sha256": zip_sha,
        "members": names,
        "zip_test": "ok",
    }
    (PRIVATE / "zip_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    public_manifest = json.loads((PUBLIC / "manifest.json").read_text(encoding="utf-8"))
    public_manifest["private_zip"] = {
        "filename": ZIP_PATH.name,
        "bytes": receipt["bytes"],
        "sha256": zip_sha,
        "contains_all_120_responses": True,
        "contains_prompts": False,
        "contains_retrieved_passages": False,
        "contains_lightrag_snapshot": False,
        "contains_lora_adapter": False,
    }
    (PUBLIC / "manifest.json").write_text(json.dumps(public_manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"zip_sha256": zip_sha, "bytes": receipt["bytes"], "safety": manifest["safety"]}, indent=2))


if __name__ == "__main__":
    main()
