"""Build the shareable Model F results package from local artifacts.

Reads saved experiment and audit files. Writes derived JSON under
``results/model-f-handoff-v1/``. Does not edit the source experiment.
Omits prompts, retrieved passages, evidence claims, and full drafts.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "model-f-handoff-v1"
EXP = ROOT / "experiments" / "EXP-20261102-001-model-f-unified-panel"
AUDIT = ROOT / "audits" / "AUDIT-20261102-001-model-f-frel-prejudge"
PILOT = ROOT / "experiments" / "EXP-model-f-quick-pilot"
STAGE = ROOT / "experiments" / "EXP-model-f-desc-vs-rel-32"

METRIC_KEYS = (
    "writing_quality",
    "task_adherence",
    "stance_consistency",
    "style_distance",
    "persona_utility",
    "claim_support",
    "word_count",
    "latency_seconds",
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dir_census(path: Path) -> dict:
    if not path.exists():
        return {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "present": False}
    files = 0
    nbytes = 0
    for item in path.rglob("*"):
        if item.is_file():
            files += 1
            nbytes += item.stat().st_size
    rel = str(path.relative_to(ROOT)).replace("\\", "/")
    return {"path": rel, "present": True, "files": files, "bytes": nbytes}


def _row(record: dict) -> dict:
    gates = {
        name: {
            "pass_rate": values.get("pass_rate"),
            "n": values.get("n"),
            "failures": values.get("failures"),
            "missing": values.get("missing"),
        }
        for name, values in record["gates"].items()
    }
    metrics = {
        name: {
            "mean": record["metrics"][name].get("mean"),
            "n": record["metrics"][name].get("n"),
            "missing": record["metrics"][name].get("missing"),
        }
        for name in METRIC_KEYS
    }
    return {
        "architecture": record["architecture"],
        "medium": record["medium"],
        "drafts": record["drafts"],
        "eligible": record["eligible"],
        "gates": gates,
        "metrics": metrics,
    }


def _example() -> dict | None:
    path = EXP / "raw" / "generations" / "gold_027__F-REL__blog.json"
    if not path.is_file():
        return None
    record = _load(path)
    answer = str(record.get("answer") or "")
    return {
        "role": "pointer to a local generation, without the draft text",
        "sharing": (
            "The GitHub repository is public. This package does not copy draft text, "
            "prompts, accepted claims, or retrieved passages. Read the local file for the draft."
        ),
        "question_id": record.get("question_id"),
        "medium": record.get("medium"),
        "arm": record.get("arm"),
        "query": record.get("query"),
        "answer_characters": len(answer),
        "seed": (record.get("generation_config") or {}).get("seed"),
        "accepted_attempt": (record.get("generation_config") or {}).get("accepted_attempt"),
        "max_new_tokens": (record.get("generation_config") or {}).get("max_new_tokens"),
        "local_path": "experiments/EXP-20261102-001-model-f-unified-panel/raw/generations/gold_027__F-REL__blog.json",
        "why_this_cell": (
            "Pre-judge audit: near the blog median, attempt 1, completion passed, "
            "and no deterministic failure on that cell. Accepted-claim strings were largely unused."
        ),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = _load(EXP / "unified_eval" / "summary.json")
    comparison = _load(EXP / "unified_eval" / "comparison.json")
    evaluation = _load(EXP / "unified_eval" / "evaluation_run.json")
    manifest = _load(EXP / "generation_manifest.json")
    plan = _load(EXP / "unified_plan.json")
    selection = _load(EXP / "posthoc_arm_selection.json")
    index = _load(EXP / "raw" / "lightrag_query" / "reports" / "indexing_summary.json")
    audit = _load(AUDIT / "summary.json")
    pilot_sel = _load(PILOT / "arm_selection.json")
    pilot_cmp = _load(PILOT / "arm_comparison.json")
    stage_rec = _load(STAGE / "desc_vs_rel_recommendation.json")
    stage_cmp = _load(STAGE / "desc_vs_rel_comparison.json")

    rows = [_row(record) for record in comparison["rows"]]
    model_f = [row for row in rows if row["architecture"] == "Model F"]
    overall = [row for row in rows if row["medium"] == "all"]

    command = evaluation.get("command") or ""
    sanitized_command = "python -m host_finetune.model_f_experiment unified-judge --profile F-REL --out experiments/EXP-20261102-001-model-f-unified-panel"
    provenance = {
        "experiment_id": "EXP-20261102-001-model-f-unified-panel",
        "profile": manifest.get("profile"),
        "protocol": manifest.get("protocol"),
        "generation_manifest_status": manifest.get("status"),
        "judge_overlay_on_generation_manifest": manifest.get("judge_overlay"),
        "note": (
            "generation_manifest.json still says generated_not_judged. "
            "unified_eval/summary.json and evaluation_run.json are the judge record."
        ),
        "gold_sha256": manifest.get("gold_sha256"),
        "generation_sha256": summary.get("generation_sha256"),
        "judge": {
            "model": evaluation.get("judge_model"),
            "config": evaluation.get("judge_config"),
            "timestamp": evaluation.get("timestamp"),
            "git_commit": (evaluation.get("git") or {}).get("commit"),
            "git_dirty": (evaluation.get("git") or {}).get("dirty"),
            "python": evaluation.get("python"),
            "recorded_command_contained_a_local_absolute_path": ":\\" in command or command.startswith("/"),
            "sanitized_command": sanitized_command,
        },
        "decoding": manifest.get("decoding"),
        "retrieval": {
            "backend": (manifest.get("retrieval") or {}).get("backend"),
            "mode": (manifest.get("retrieval") or {}).get("mode"),
            "evidence_budget": (manifest.get("retrieval") or {}).get("evidence_budget"),
            "top_k": (manifest.get("retrieval") or {}).get("top_k"),
            "num_ctx": (manifest.get("retrieval") or {}).get("num_ctx"),
            "live_store_queried": (manifest.get("retrieval") or {}).get("live_store_queried"),
            "snapshot": "artifacts/lightrag/model_f/storage_expand_v1_snapshot",
        },
        "model": {
            "base_model": manifest.get("base_model"),
            "adapter": "host_finetune/output/lemkin_lora_relabel",
            "quantization_requested": manifest.get("quantization"),
            "max_seq_length": manifest.get("max_seq_length"),
            "model_load_as_recorded": manifest.get("model_load"),
        },
        "graph": {
            "nodes": index.get("graph_nodes"),
            "edges": index.get("graph_edges"),
            "graph_count_source": index.get("graph_count_source"),
            "successful_chunks": index.get("successful_chunks"),
            "completed_graph_indexed": index.get("completed_graph_indexed"),
            "lightrag_version": (index.get("configuration") or {}).get("lightrag_version"),
            "extraction_model": (index.get("configuration") or {}).get("extraction_model"),
            "embed_model": (index.get("configuration") or {}).get("embed_model"),
            "embedding_dim": (index.get("configuration") or {}).get("embedding_dim"),
            "num_ctx": (index.get("configuration") or {}).get("num_ctx"),
        },
        "plan_status": plan.get("status"),
        "posthoc_selection": selection,
        "pilot": {
            "experiment_id": pilot_sel.get("experiment_id"),
            "selected_arm": pilot_sel.get("selected_arm"),
            "frozen": pilot_sel.get("frozen"),
            "status": pilot_sel.get("status"),
            "n_complete_questions": len(pilot_sel.get("complete_question_ids") or []),
            "arms": {
                name: {
                    "n": values.get("n"),
                    "factual_pass_rate": values.get("factual_pass_rate"),
                    "copy_or_leak_failure_rate": values.get("copy_or_leak_failure_rate"),
                }
                for name, values in (pilot_cmp.get("arms") or {}).items()
            },
        },
        "stage2": {
            "experiment_id": stage_rec.get("experiment_id"),
            "selected_arm": stage_rec.get("selected_arm"),
            "status": stage_rec.get("status"),
            "compared_questions": stage_rec.get("compared_questions"),
            "question_wins": stage_rec.get("question_wins"),
            "question_ties": stage_rec.get("question_ties"),
            "arms": {
                name: {
                    "n": values.get("n"),
                    "echo_adjusted_pass_rate": values.get("echo_adjusted_pass_rate"),
                    "copy_or_leak_failure_rate": values.get("copy_or_leak_failure_rate"),
                    "scorer_faithfulness_pass_rate": values.get("scorer_faithfulness_pass_rate"),
                    "numeric_failure_rate": values.get("numeric_failure_rate"),
                }
                for name, values in (stage_cmp.get("arms") or {}).items()
            },
        },
        "hardware_gpu_name_in_generation_manifest": False,
    }

    audit_summary = {
        "audit_id": "AUDIT-20261102-001-model-f-frel-prejudge",
        "source": "audits/AUDIT-20261102-001-model-f-frel-prejudge/summary.json",
        "manifest_status_at_audit": audit.get("manifest_status"),
        "saved_generation_files": audit.get("saved_generation_files"),
        "overall": audit.get("overall"),
        "training_copy_note": (
            "summary.json records training_copy_fail count 0 with denominator 104. "
            "The audit report, after re-reading unified_eval/rows.jsonl, states 0 of 120. "
            "Both are recorded. This export does not recompute the gate."
        ),
    }

    excluded = {
        "policy": "Raw experimental files were not edited. These paths are omitted from Git.",
        "directories": [
            _dir_census(EXP),
            _dir_census(EXP / "raw"),
            _dir_census(EXP / "unified_eval"),
            _dir_census(PILOT),
            _dir_census(STAGE),
            _dir_census(ROOT / "experiments" / "EXP-20261102-004-model-f-frel-smoke"),
            _dir_census(ROOT / "artifacts" / "lightrag" / "model_f" / "storage_expand_v1_snapshot"),
            _dir_census(ROOT / "host_finetune" / "output" / "lemkin_lora_relabel"),
        ],
        "file_hashes": {
            rel: _sha256(ROOT / rel)
            for rel in (
                "experiments/EXP-20261102-001-model-f-unified-panel/unified_eval/summary.json",
                "experiments/EXP-20261102-001-model-f-unified-panel/unified_eval/comparison.json",
                "experiments/EXP-20261102-001-model-f-unified-panel/unified_eval/evaluation_run.json",
                "experiments/EXP-20261102-001-model-f-unified-panel/generation_manifest.json",
                "experiments/EXP-20261102-001-model-f-unified-panel/posthoc_arm_selection.json",
                "experiments/EXP-20261102-001-model-f-unified-panel/raw/lightrag_query/reports/indexing_summary.json",
                "experiments/EXP-20261102-001-model-f-unified-panel/raw/lightrag_query/snapshot_hashes.json",
                "audits/AUDIT-20261102-001-model-f-frel-prejudge/summary.json",
                "audits/AUDIT-20261102-001-model-f-frel-prejudge/REPORT.md",
            )
        },
        "recorded_transfer_hashes": {
            "source": "docs/handoff/MODEL_F_DEPLOYMENT_TRANSFER_20261102.md",
            "snapshot_zip_sha256": "4e7a00a977efe0c4ab121b0356704ecd4472cb17d0a96533f8c80a5ad500b9ea",
            "snapshot_zip_bytes": 481019443,
            "adapter_zip_sha256": "644041cbe47e0a1ce69997e790be799b5ed42862be2854b792e94a24dd93777f",
            "adapter_weights_sha256": "850e9aa88d04fa4e7c9490d5d974715c8a1018c47c7ada340d642b4b2afea530",
            "graph_audit_sha256": "2672aee28b5564876f6f8d547745acdc3eda7780e6b5df5d4812f5def2e0b4dd",
            "rehashed_in_this_export": False,
        },
    }

    documents = {
        "aggregate_summary.json": {
            "schema": summary.get("schema"),
            "judge_model": summary.get("judge_model"),
            "temperature": summary.get("temperature"),
            "blinded": summary.get("blinded"),
            "n_traces": summary.get("n_traces"),
            "generation_sha256": summary.get("generation_sha256"),
            "utility_weights": summary.get("utility_weights"),
            "writing_quality_calibration": summary.get("writing_quality_calibration"),
            "stance_calibration": summary.get("stance_calibration"),
            "model": summary.get("model"),
        },
        "architecture_comparison.json": {"medium": "all", "rows": overall},
        "per_medium.json": {"architecture": "Model F", "rows": model_f},
        "deterministic_audit_summary.json": audit_summary,
        "provenance.json": provenance,
        "examples.json": {
            "included": [_example()] if _example() else [],
            "withheld": [
                "Draft text, including cells the audit called representative",
                "Full prompts and source-grounded claim text",
                "Retrieved passages and LightRAG vector stores",
                "Cells flagged for a 12-word source copy",
                "Tracking URLs that appear in off-topic drafts",
            ],
        },
        "excluded_artifacts.json": excluded,
    }
    for name, payload in documents.items():
        text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        (OUT / name).write_text(text, encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
