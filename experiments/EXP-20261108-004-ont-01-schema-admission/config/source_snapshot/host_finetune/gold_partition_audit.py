"""CPU-only gold-family audit. Never opens Chroma clients or calls models.

Only writes a new registered TASK-002 directory; source bodies stay in memory.
The existing overlap predicates and train selection are reused unchanged.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import redirect_stdout
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
from importlib import metadata
import json
import platform
from pathlib import Path
import sqlite3
import subprocess
import sys

from host_finetune.split_groups import (
    assert_manifest_matches, file_sha256, indices_for_split, normalize_overlap_text,
)
from host_finetune.train_only_index import (
    ROOT, DEFAULT_ASSIGNMENTS, DEFAULT_DATASET, DEFAULT_GOLD_DISPOSITIONS,
    DEFAULT_GOLD_EVAL, DEFAULT_PERSIST_DIR, TRAIN_ONLY_COLLECTION,
    assignment_rows, blocked_group_ids, gold_field_values, headline_overlap_groups,
    load_jsonl, matching_gold_fields, select_train_documents,
)

EXPERIMENT_ID = "EXP-20261108-001-gold-partition-audit"
OUT = ROOT / "experiments" / EXPERIMENT_ID
CORPUS = ROOT / "experiments/EXP-20261004-009-dense-rag-v1/metrics/corpus.json"
# Manually paraphrased development questions, anchored to existing train rows.
# No answer, expected fact, or source body is exported. These are proposals only.
QUESTION_SEEDS = (
    (718, "How should a SaaS team hand a customer account to a new owner?"),
    (787, "When should a young SaaS company start pursuing enterprise buyers?"),
    (812, "What should founders consider before sharing the CEO role?"),
    (813, "Which step deserves the most attention when trying to close a sale?"),
    (1014, "How should founders think about their ownership by the time of an exit?"),
    (1077, "Which habits distinguish exceptional sales representatives?"),
    (1259, "How should a founder handle a sales representative who misled customers?"),
    (1446, "What are the tradeoffs of releasing a SaaS product very early?"),
    (1669, "When is it useful for both founders to attend an investor pitch?"),
    (1732, "How can sales managers keep their strongest performers?"),
    (1972, "How should a SaaS team approach winning back customers who left?"),
    (2298, "How can a cofounder decide whether it is time to leave the company?"),
)

REVIEW_ID = "EXP-20261108-003-gold-exposure-review"
TRAIN_RUN = ROOT / "experiments/EXP-20261003-005-relabel-qlora"
TRAINER_STATE = ROOT / "host_finetune/output/lemkin_lora_relabel/checkpoint-2674/trainer_state.json"
MORE_QUESTIONS = (
    (3977, "When does increasing prices fail to address the underlying cause of slow growth?"),
    (8068, "Under what conditions should a mid-market SaaS vendor build a customer-requested feature?"),
    (8349, "Who should choose a young SaaS company's CRM, and why?"),
    (8470, "How can a startup relate an advisor's equity grant to the value and timing of their work?"),
    (8534, "Which early customer and growth signals suggest product-market fit?"),
    (8925, "How can a SaaS founder recognize an emerging brand within a narrow market?"),
    (9902, "Which technical experiences should founders seek in engineering and product leaders?"),
    (10250, "How can a SaaS founder size a cash reserve for unexpected setbacks?"),
    (10983, "How should a SaaS founder weigh customer concentration against opportunities to grow?"),
    (11325, "How can an early SaaS team estimate customer lifetime value before it has enough history?"),
    (8205, "What does the author's customer-visit example suggest, and what causal claims would go beyond it?"),
    (10931, "How can company scale and customer size affect the share of new revenue from existing accounts?"),
)
# Manual review labels, not extracted ontology types or benchmark answer keys.
QUESTION_REVIEW = (
    ("customer success", "account transition; action and expected benefit"),
    ("enterprise readiness", "commitment and delivery-time condition"),
    ("founder governance", "authority risk and outlier exceptions"),
    ("sales execution", "response behavior and deal progression"),
    ("founder ownership", "funding context and dilution; source-era estimates"),
    ("sales productivity", "time allocation and performance distinction"),
    ("sales governance", "misrepresentation and control processes; attribute author's stance"),
    ("product launch", "timing tradeoff and first-person experience"),
    ("fundraising process", "founder role and complementary expertise"),
    ("sales leadership", "retention actions and limits"),
    ("customer reactivation", "campaign action, cadence, and time horizon"),
    ("founder sustainability", "decision process and fatigue; thematic proximity to gold_020"),
    ("pricing strategy", "price action versus acquisition mechanism"),
    ("product roadmap", "conjunctive inclusion and exclusion conditions"),
    ("tool ownership", "executive decision authority and tool choice"),
    ("advisor compensation", "valuation assumptions, formula, and vesting"),
    ("product-market fit", "metric thresholds and qualified interpretation"),
    ("brand development", "market segment, customer base, and referrals"),
    ("technical hiring", "role-specific experience requirements"),
    ("cash planning", "reserve-to-ARR relation and anecdotal limits"),
    ("customer concentration", "risk, cohort size, and investor exceptions"),
    ("unit economics", "formula, comparable selection, and uncertainty"),
    ("customer engagement", "observed association versus causal inference"),
    ("revenue expansion", "customer segment and company-scale conditions"),
)


def digest_json(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def audit_selection(dataset: list[dict], assignments: list[dict],
                    dispositions: list[dict], gold: list[dict]) -> tuple[dict, list[dict], set[str]]:
    """Replay exclusion, retaining metadata-only overlap details for all splits."""
    family = blocked_group_ids(assignments, dispositions)
    family_docs, family_stats = select_train_documents(dataset, assignments, family)
    overlaps, details = headline_overlap_groups(dataset, assignments, gold, family)
    blocked = family | overlaps
    selected, final_stats = select_train_documents(dataset, assignments, blocked)
    records = assignment_rows(assignments)
    report = {
        "splits": dict(Counter(r["split"] for r in records)),
        "recorded_blocked_family_count": len(family),
        "family_only_selection": family_stats,
        "final_selection": final_stats,
        "additional_overlap_families": details,
        "removed_after_headline_check": sorted({d["id"] for d in family_docs} - {d["id"] for d in selected}),
        "overlap_rows_by_split": dict(Counter(r["split"] for r in records if r["group_id"] in blocked)),
        "remaining_selected_overlap_families": sorted({d["metadata"]["group_id"] for d in selected} & blocked),
        "selected_membership_sha256": digest_json(sorted(d["id"] for d in selected)),
    }
    return report, selected, blocked


def read_index(path: Path) -> tuple[list[dict], dict]:
    """Read only the target collection's metadata segment; no vector operations."""
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as conn:
        conn.execute("PRAGMA query_only=ON")
        collection = conn.execute("SELECT id, schema_str FROM collections WHERE name=?", (TRAIN_ONLY_COLLECTION,)).fetchall()
        if len(collection) != 1:
            raise ValueError("expected exactly one lemkin_train_only collection")
        collection_id, schema_text = collection[0]
        segments = conn.execute("SELECT id FROM segments WHERE collection=? AND scope='METADATA'", (collection_id,)).fetchall()
        if len(segments) != 1:
            raise ValueError("expected exactly one metadata segment")
        rows = {}
        for doc_id, in conn.execute("SELECT embedding_id FROM embeddings WHERE segment_id=?", (segments[0][0],)):
            rows[doc_id] = {}
        for doc_id, key, string, integer in conn.execute(
            "SELECT e.embedding_id,m.key,m.string_value,m.int_value FROM embeddings e "
            "JOIN embedding_metadata m ON e.id=m.id WHERE e.segment_id=?", (segments[0][0],)
        ):
            if key == "chroma:document":
                rows[doc_id]["text"] = string
            elif key in ("group_id", "medium", "split", "row_id"):
                rows[doc_id][key] = integer if key == "row_id" else string
        schema = json.loads(schema_text or "{}")
        space = schema.get("keys", {}).get("#embedding", {}).get("float_list", {}).get("vector_index", {}).get("config", {}).get("space")
    return [{"id": k, "text": v.get("text") or "", "metadata": {f: v.get(f) for f in ("group_id", "medium", "split", "row_id")}} for k, v in sorted(rows.items())], {"collection_id": collection_id, "metadata_segment_id": segments[0][0], "space": space}


def compare_index(expected: list[dict], persisted: list[dict], blocked: set[str], gold: list[dict]) -> dict:
    wanted = {d["id"]: d for d in expected}
    actual = {d["id"]: d for d in persisted}
    # Titles come from frozen assignments; equality below verifies the persisted
    # bodies and metadata. Scan actual bodies as well to expose unexpected rows.
    # Reuse the batched matcher: it normalizes each document and gold field
    # once, unlike calling the single-document matcher for every field/row.
    _, details = headline_overlap_groups(
        [{"output": d["text"]} for d in persisted],
        [{"row_index": i, "group_id": d["id"]} for i, d in enumerate(persisted)], gold,
    )
    hits = [{"id": d["group_id"], "hits": d["hits"]} for d in details]
    return {
        "document_count": len(persisted),
        "train_26148_present": "train_26148" in actual,
        "missing_ids": sorted(wanted.keys() - actual.keys()),
        "unexpected_ids": sorted(actual.keys() - wanted.keys()),
        "content_or_metadata_mismatch_ids": sorted(k for k in wanted.keys() & actual.keys() if wanted[k] != actual[k]),
        "non_train_ids": sorted(d["id"] for d in persisted if d["metadata"].get("split") != "train"),
        "blocked_family_ids": sorted(d["id"] for d in persisted if d["metadata"].get("group_id") in blocked),
        "remaining_lexical_overlaps": hits,
        "logical_content_sha256": digest_json(sorted(persisted, key=lambda d: d["id"])),
    }


def development_questions(dataset: list[dict], assignments: list[dict], selected: list[dict],
                          blocked: set[str], gold: list[dict], seeds=QUESTION_SEEDS) -> tuple[list[dict], list[dict]]:
    records = assignment_rows(assignments)
    allowed = {d["metadata"]["row_id"] for d in selected}
    fields = gold_field_values(gold)
    accepted, rejected = [], []
    groups = set()
    for row_id, question in seeds:
        if not 0 <= row_id < len(records):
            rejected.append({"row_id": row_id, "reason": "missing source row"})
            continue
        rec = records[row_id]
        reason = None
        if row_id not in allowed or rec["group_id"] in blocked:
            reason = "excluded family or unselected row"
        elif rec.get("source_platform") != "blog" or rec.get("sft_role") != "unchanged":
            reason = "requires unchanged train blog"
        elif rec["group_id"] in groups:
            reason = "duplicate proposed family"
        elif matching_gold_fields(question, question, fields) or matching_gold_fields(
            dataset[row_id].get("output", ""), rec.get("base_title", ""), fields
        ):
            reason = "gold lexical overlap"
        if reason:
            rejected.append({"row_id": row_id, "reason": reason})
            continue
        groups.add(rec["group_id"])
        accepted.append({
            "id": f"dev_candidate_{len(accepted) + 1:03d}", "status": "PROPOSED",
            "question": question, "source_row_id": row_id, "group_id": rec["group_id"],
            "source_file": rec["source_file"], "source_line": rec["source_line"],
            "split": rec["split"], "medium": "blog",
            "source_output_sha256": hashlib.sha256(dataset[row_id]["output"].encode()).hexdigest(),
            "use": "development only; source is already in training and retrieval; not an independent test",
        })
    if {r["id"] for r in accepted} & {r["id"] for r in gold}:
        raise ValueError("development and gold IDs collide")
    return accepted, rejected


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def replay_training_loader(dataset_path: Path, trainer_path: Path) -> list[dict]:
    """Execute the actual pure loader body without importing the GPU trainer.

    Dataset.from_list is an identity adapter here. No SFTTrainer/model is loaded.
    This preserves the loader's blank-line and malformed-JSON behavior.
    """
    tree = ast.parse(trainer_path.read_text(encoding="utf-8"))
    loader = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "load_dataset")
    class RowsOnly:
        @staticmethod
        def from_list(rows):
            return rows
    namespace = {"Path": Path, "Dataset": RowsOnly, "json": json}
    exec(compile(ast.Module(body=[loader], type_ignores=[]), str(trainer_path), "exec"), namespace)
    with redirect_stdout(io.StringIO()):
        return namespace["load_dataset"](dataset_path)


def read_training_log(path: Path) -> str:
    """PowerShell logs can be UTF-16; decode strictly rather than hiding errors."""
    raw = path.read_bytes()
    return raw.decode("utf-16" if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig")


def saved_training_arguments(path: Path) -> dict:
    """Inspect whitelisted scalar pickle opcodes without unpickling/execution."""
    import pickletools
    import zipfile
    keys = {"num_train_epochs", "max_steps", "dataloader_drop_last", "train_sampling_strategy",
            "packing", "max_length", "dataset_text_field", "gradient_accumulation_steps",
            "per_device_train_batch_size", "seed", "load_best_model_at_end",
            "assistant_only_loss", "completion_only_loss"}
    with zipfile.ZipFile(path) as archive:
        name = next(n for n in archive.namelist() if n.endswith("/data.pkl"))
        operations = list(pickletools.genops(archive.read(name)))
    result = {}
    for i, (_, key, _) in enumerate(operations):
        if not isinstance(key, str) or key not in keys:
            continue
        for opcode, value, _ in operations[i + 1:]:
            if opcode.name in ("BINPUT", "LONG_BINPUT", "MEMOIZE", "PUT"):
                continue
            if opcode.name in ("BININT", "BININT1", "BININT2", "INT", "LONG1", "LONG4", "BINFLOAT", "BINUNICODE", "SHORT_BINUNICODE", "UNICODE"):
                result[key] = value
            elif opcode.name in ("NEWTRUE", "NEWFALSE", "NONE"):
                result[key] = {"NEWTRUE": True, "NEWFALSE": False, "NONE": None}[opcode.name]
            break
    return result


def literal_gold_matches(dataset: list[dict], records: list[dict], gold: list[dict]) -> list[dict]:
    """Literal, case-sensitive complete gold-field substrings; no fuzzy labeling.

    Topic is the benchmark question/topic text, not the complete rendered
    generation prompt. Prompt/input occurrences are separated from target text.
    """
    fields = [(gid, field, value, hashlib.sha256(value.encode()).hexdigest())
              for gid, field, value in gold_field_values(gold)]
    hits = []
    for row, rec in zip(dataset, records):
        for dataset_field in ("instruction", "input", "output"):
            text = row.get(dataset_field) or ""
            if not text:
                continue
            for gid, field, value, digest in fields:
                if value in text:
                    hits.append({"row_id": rec["row_index"], "split": rec["split"],
                                 "gold_id": gid, "gold_field": field, "dataset_field": dataset_field,
                                 "gold_value_sha256": digest, "entire_field_equal": text == value})
    return hits


def training_exposure(dataset: list[dict], records: list[dict], blocked: set[str],
                      gold: list[dict], manifest: dict, trainer_state: dict, log: str,
                      saved_arguments: dict | None = None) -> dict:
    """Separate selected membership, literal text evidence, and run inference."""
    from host_finetune.compare_adapters import MEDIUMS, prompt_for
    from host_finetune.llama_chat_format import user_message_from_alpaca_row

    train_ids = set(indices_for_split(records, "train"))
    validation_ids = set(indices_for_split(records, "validation"))
    fields = gold_field_values(gold)
    overlap_rows = []
    for rec in records:
        if rec["group_id"] not in blocked:
            continue
        i = rec["row_index"]
        overlap_rows.append({
            "row_id": i, "split": rec["split"], "group_id": rec["group_id"],
            "source_file": rec["source_file"], "source_line": rec["source_line"],
            "included_in_train_dataset": i in train_ids,
            "included_in_validation_dataset": i in validation_ids,
            "sft_role": rec.get("sft_role"),
            "direct_output_or_title_lexical_hits": matching_gold_fields(
                dataset[i].get("output", ""), rec.get("base_title", ""), fields),
        })
    literal = literal_gold_matches(dataset, records, gold)
    prompt_lookup = {}
    for row in gold:
        for medium in MEDIUMS:
            prompt_lookup.setdefault(prompt_for(medium, row["topic"]), []).append((row["id"], medium["medium"]))
    exact_prompts = []
    for row, rec in zip(dataset, records):
        user_text = user_message_from_alpaca_row(row.get("instruction", ""), row.get("input", ""))
        for gold_id, medium in prompt_lookup.get(user_text, []):
            exact_prompts.append({"row_id": rec["row_index"], "split": rec["split"], "gold_id": gold_id, "medium": medium})
    counts = {}
    for split in ("train", "validation", "test"):
        counts[split] = {
            "selected_rows": len(indices_for_split(records, split)),
            "gold_family_rows": sum(r["split"] == split for r in overlap_rows),
            "family_rows_with_direct_output_or_title_match": sum(r["split"] == split and bool(r["direct_output_or_title_lexical_hits"]) for r in overlap_rows),
            "family_rows_without_direct_output_or_title_match": sum(r["split"] == split and not r["direct_output_or_title_lexical_hits"] for r in overlap_rows),
            "literal_topic_in_instruction_rows": len({h["row_id"] for h in literal if h["split"] == split and h["gold_field"] == "topic" and h["dataset_field"] == "instruction"}),
            "literal_topic_in_output_rows": len({h["row_id"] for h in literal if h["split"] == split and h["gold_field"] == "topic" and h["dataset_field"] == "output"}),
            "literal_complete_reference_in_output_rows": len({h["row_id"] for h in literal if h["split"] == split and h["gold_field"] == "gold_reference" and h["dataset_field"] == "output"}),
            "literal_topic_gold_ids": sorted({h["gold_id"] for h in literal if h["split"] == split and h["gold_field"] == "topic"}),
            "exact_retrieval_off_user_message_rows": len({h["row_id"] for h in exact_prompts if h["split"] == split}),
        }
    full_run = (trainer_state.get("epoch") == 2.0 and trainer_state.get("global_step") == 2674
                and trainer_state.get("max_steps") == 2674 and trainer_state.get("num_train_epochs") == 2
                and "Num examples = 21,377" in log and "train=21377 eval=2588" in log
                and "Saving LoRA adapter ->" in log and "FINETUNE_TRAIN_HEAD_N=" not in log)
    if saved_arguments is not None:
        full_run = full_run and all(saved_arguments.get(k) == v for k, v in {
            "num_train_epochs": 2, "max_steps": -1, "per_device_train_batch_size": 1,
            "dataloader_drop_last": False, "train_sampling_strategy": "random", "packing": False,
            "gradient_accumulation_steps": 16, "max_length": 512, "seed": 42,
        }.items())
    return {
        "counts": counts, "overlap_rows": overlap_rows, "literal_gold_field_matches": literal,
        "exact_current_retrieval_off_user_message_matches": exact_prompts,
        "train_membership_sha256": digest_json(sorted(train_ids)),
        "training_manifest_dataset_hash": manifest["data"]["processed_dataset_hash"],
        "completion_evidence_consistent": full_run,
        "checkpoint_summary": {k: trainer_state.get(k) for k in (
            "epoch", "global_step", "max_steps", "num_train_epochs", "train_batch_size", "best_global_step", "num_input_tokens_seen")},
        "saved_training_arguments_safe_scalar_inspection": saved_arguments,
        "masking_note": "TRL assistant_only_loss=False in saved config is distinct from the explicit Unsloth train_on_responses_only wrapper in finetune.py. That wrapper supplies assistant-only labels.",
        "consumption_conclusion": (
            f"{counts['train']['gold_family_rows']} family-overlap rows are verified members of the actual train dataset. "
            "Completed two-epoch logs and checkpoint state strongly support consumption during training; "
            "there is no row-level batch ledger, so individual visits, effective loss tokens, and weight influence are not independently verified."
            if full_run else "Run completion evidence inconsistent or incomplete; do not infer complete consumption."),
        "validation_conclusion": "16 family rows enter validation loss and checkpoint selection, not train gradients. Four test-family rows are excluded from both trainer datasets.",
        "interpretation": "Author-corpus familiarity is intended. Family overlap is not automatically invalid leakage. Literal benchmark-topic exposure and reused topics limit unseen-question/independent-generalization claims; they do not by themselves invalidate controlled same-benchmark comparisons or justify retraining.",
        "limitations": [
            "No per-example training ledger or historical tokenized dataset was persisted in the registered run.",
            "Assistant-only loss masks instruction labels; prompt text still conditions target prediction. Source/target presence does not prove memorization.",
            "512-token truncation and response masking may affect token-level exposure. This audit does not reconstruct historical token tensors. Only whitelisted scalar opcodes in training_args.bin are inspected; no pickle is executed.",
            "Exact topic substrings and exact current retrieval-off user-message equality are reported separately. This does not establish equality of complete chat token sequences or retrieval-on prompts. Expected-fact/reference matches are separately labeled.",
            "Lexical family/near-duplicate detection does not certify semantic independence or quantify paraphrase similarity.",
            "Checkpoint metadata corroborates the registered log but is outside that historical experiment; no model weights or optimizer state were read.",
            "EXP-005 was dirty and did not record runtime code hashes or a full package lock. Current trainer/config/chat code matches its recorded commit, but exact historical runtime code and dependency identity cannot be proved from that alone.",
        ],
    }


def review_candidates(candidates: list[dict], dataset: list[dict]) -> list[dict]:
    """Attach source locators and manual review notes, without body spans."""
    reviewed = []
    source_cache = {}
    for candidate, (topic, purpose) in zip(candidates, QUESTION_REVIEW):
        filename = candidate["source_file"]
        if filename not in source_cache:
            source_cache[filename] = load_jsonl(ROOT / "data/cleaned" / filename)
        source = source_cache[filename][int(candidate["source_line"]) - 1]
        output = dataset[candidate["source_row_id"]]["output"]
        source_text = str(source.get("title", "")) + "\n" + str(source.get("content", ""))
        reviewed.append({**candidate, "topic_area": topic, "review_focus": purpose,
                         "evidence_available": bool(output.strip()) and bool(source.get("content")),
                         "cleaned_source_path": "data/cleaned/" + filename,
                         "source_url": source.get("url"), "source_date": source.get("date"),
                         "cleaned_record_sha256": digest_json(source),
                         "source_body_contains_output_normalized": normalize_overlap_text(output) in normalize_overlap_text(source_text),
                         "review_status": "source reviewed; suitable proposed ontology-development question; no answer key or scored retrieval claim"})
    return reviewed


def write_yaml(path: Path, mapping: dict) -> None:
    def lines(obj: dict, indent=0):
        result = []
        for key, value in obj.items():
            result.append(" " * indent + key + ":" + ("" if isinstance(value, dict) else " " + json.dumps(value)))
            if isinstance(value, dict):
                result.extend(lines(value, indent + 2))
        return result
    path.write_text("\n".join(lines(mapping)) + "\n", encoding="utf-8")


def prepare_coordination_proposal(out: Path) -> dict:
    """Write exact owner-review edits and simulate them without changing live state."""
    import copy
    import difflib
    from unittest.mock import patch
    from project_pm import check as pm

    state_path = ROOT / "docs/project/state.json"
    tasks_path = ROOT / "docs/project/tasks.json"
    handoff_path = ROOT / "docs/project/HANDOFF.md"
    before = {p: p.read_text(encoding="utf-8") for p in (state_path, tasks_path, handoff_path)}
    state, tasks = json.loads(before[state_path]), json.loads(before[tasks_path])
    task = next(t for t in tasks["tasks"] if t["id"] == "TASK-002")
    for exp in ("EXP-20261108-002-gold-exposure-review", REVIEW_ID):
        owned = f"experiments/{exp}/"
        if owned not in task["owns_files"]:
            task["owns_files"].append(owned)
    # A done task's owned files are not accepted by _check_git. Record precisely
    # its still-uncommitted paths via the existing known_dirty_prefixes field.
    # No broad prefix is added and no validator code/rule is changed.
    task["status"] = "done"
    task["verification"] = (
        f"2026-11-08: EXP-001 audit preserved; corrected {REVIEW_ID} audit checks passed, "
        "training input/split hashes match EXP-005, 184/16/4 family-row split mapping checked, "
        "24 development candidates source-reviewed, and 21,193-document index parity passed. "
        "Unit-test, manifest, and proposed-state verification: experiments/" + REVIEW_ID + "/raw/verification.json. "
        "Owner must apply this proposed snapshot and rerun the full live checker before accepting closure."
    )
    task["files"] = list(task["owns_files"])
    state["updated"] = "2026-11-08"
    state["git"]["branch"] = git("branch", "--show-current")
    state["git"]["commit"] = git("rev-parse", "HEAD")
    state["git"]["worktree"] = "dirty"
    for owned in task["owns_files"]:
        if owned not in state["git"]["known_dirty_prefixes"]:
            state["git"]["known_dirty_prefixes"].append(owned)
    state["phase"] = "ready-for-ontology-planning"
    state["next_task"] = "TASK-003"
    state["evaluation"]["leakage_audit"] = (
        f"TASK-002: {REVIEW_ID}. Verified 184 train, 16 validation, 4 test gold-family rows; "
        "184 are actual training-dataset members, with completed two-epoch consumption strongly supported but no per-row ledger. "
        "24 proposed train-source development questions; exact gold-family exclusions retained in the 21,193-document index. "
        "Gold remains a reused comparison benchmark, not an independent final test."
    )
    for evidence in (f"experiments/{REVIEW_ID}/manifest.yaml", f"experiments/{REVIEW_ID}/notes.md"):
        if evidence not in state["evidence"]:
            state["evidence"].append(evidence)
    next_action = "Begin TASK-003 planning only in docs/architecture/ONTOLOGY_PLAN.md on its assigned branch when the owner starts it. Do not implement ontology extraction or alter frozen data or indexes."
    handoff = before[handoff_path]
    replacements = {
        "updated": state["updated"], "git_commit": state["git"]["commit"],
        "git_branch": state["git"]["branch"], "active_task": "TASK-003", "session": "active",
        "next_action": next_action,
    }
    for key, value in replacements.items():
        handoff = "\n".join(f"- {key}: {value}" if line.startswith(f"- {key}:") else line for line in handoff.split("\n"))
    handoff = handoff.replace("Phase is `ready-for-parallel-prep`.", "Phase is `ready-for-ontology-planning`.")
    handoff = handoff.replace(
        "Codex, TASK-002, status proposed:",
        "Codex, TASK-002, status done after owner applies this closure snapshot and verifies the live checker:",
    )
    handoff = handoff.replace(
        "Read-only against existing indexes and splits. Do not rerun historical experiments.",
        "Also owns `experiments/EXP-20261108-002-gold-exposure-review/` (incomplete UTF-16 parsing attempt) and "
        f"`experiments/{REVIEW_ID}/` (corrected follow-up). See that run's notes and verification. "
        "Read-only against existing indexes and splits. Do not rerun historical experiments.",
    )
    handoff += (
        "\n## TASK-002 closure evidence\n\n"
        f"Corrected follow-up: `experiments/{REVIEW_ID}/notes.md`. "
        "184 gold-family train rows were selected by the historical trainer; completed two-epoch consumption is strongly supported, "
        "without per-row proof. Literal topic exposure limits independent-question claims; it is not automatic invalid leakage. "
        "The 24 development questions are proposed train-source material. Retrieval parity is unchanged. "
        "No retraining or split change is authorized. TASK-003 remains planning only.\n"
    )
    after = {state_path: json.dumps(state, indent=2) + "\n", tasks_path: json.dumps(tasks, indent=2) + "\n", handoff_path: handoff}
    proposed_dir = out / "config/proposed-project"
    proposed_dir.mkdir(parents=True, exist_ok=False)
    changes = []
    diff = []
    for path, content in after.items():
        (proposed_dir / path.name).write_text(content, encoding="utf-8")
        relative = path.relative_to(ROOT).as_posix()
        diff.extend(difflib.unified_diff(before[path].splitlines(True), content.splitlines(True), fromfile="a/" + relative, tofile="b/" + relative))
        changes.append({"path": relative, "required_current_sha256": file_sha256(path), "proposed_content_sha256": hashlib.sha256(content.encode()).hexdigest()})
    (out / "config/coordination-proposal.patch").write_text("".join(diff), encoding="utf-8")
    original_read_json = pm._read_json
    def proposed_json(path, report, code):
        if Path(path) == state_path:
            return copy.deepcopy(state)
        if Path(path) == tasks_path:
            return copy.deepcopy(tasks)
        return original_read_json(path, report, code)
    # Only the candidate file contents are substituted; the real validator,
    # live branch, worktree list, manifests, and reference checks still run.
    with patch.object(pm, "_read_json", proposed_json), patch.object(pm, "_read_handoff", return_value={k: v for k, v in replacements.items()}):
        simulated = pm.check_repository(ROOT, live_git=True).to_dict()
    if any(path.read_text(encoding="utf-8") != content for path, content in before.items()):
        raise RuntimeError("live shared state unexpectedly changed during proposal preparation")
    result = {"approval_required": True, "live_docs_edited": False, "changes": changes,
              "proposed_state_full_checker_dry_run": simulated,
              "procedure": "Owner reviews patch and current hashes; after approval, apply all three files together, then run the full checker and live-state test. Recheck concurrent Cursor changes before applying. No branch switch, commits, or validator weakening.",
              "known_dirty_rationale": "Only the five exact TASK-002-owned paths are registered because done-task ownership is not used by the live checker and the user prohibited commits. Existing unrelated dirty prefixes remain unchanged."}
    write_json(out / "raw/coordination_proposal.json", result)
    return result


def run_review() -> int:
    out = ROOT / "experiments" / REVIEW_ID
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    trainer = ROOT / "host_finetune/finetune.py"
    snapshot_paths = [DEFAULT_DATASET, DEFAULT_ASSIGNMENTS, DEFAULT_GOLD_DISPOSITIONS, DEFAULT_GOLD_EVAL,
                      TRAIN_RUN / "manifest.yaml", TRAIN_RUN / "config/commands.txt", TRAIN_RUN / "raw/train.log",
                      TRAINER_STATE, trainer, ROOT / "host_finetune/config.py", ROOT / "host_finetune/split_groups.py",
                      TRAINER_STATE.parent / "training_args.bin",
                      ROOT / "host_finetune/llama_chat_format.py", ROOT / "host_finetune/relabel_continuations.py",
                      ROOT / "host_finetune/best_metric_checkpoints.py", Path(__file__), ROOT / "tests/test_gold_partition_audit.py",
                      ROOT / "host_finetune/compare_adapters.py",
                      DEFAULT_PERSIST_DIR / "chroma.sqlite3", ROOT / "data/cleaned/jasonlemkin_blog.jsonl",
                      ROOT / "data/jasonlemkin_blog.jsonl", ROOT / "project_pm/check.py",
                      ROOT / "docs/project/state.json", ROOT / "docs/project/tasks.json", ROOT / "docs/project/HANDOFF.md"]
    # Preserve exact bytes of the completed first audit and the historical run.
    snapshot_paths += [p for p in OUT.rglob("*") if p.is_file()]
    snapshot_paths += [p for p in (ROOT / "experiments/EXP-20261108-002-gold-exposure-review").rglob("*") if p.is_file()]
    snapshot_paths += [p for p in TRAIN_RUN.rglob("*") if p.is_file()]
    snapshot_paths += [ROOT / p for p in git("diff", "--name-only").splitlines()
                       if (ROOT / p).is_file() and p not in ("host_finetune/gold_partition_audit.py", "tests/test_gold_partition_audit.py")]
    hashes = {p.relative_to(ROOT).as_posix(): file_sha256(p) for p in snapshot_paths}
    timestamp = datetime.now(timezone.utc).isoformat()
    revision = {"commit": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"), "dirty": bool(git("status", "--porcelain"))}
    training_manifest = json.loads((TRAIN_RUN / "manifest.yaml").read_text(encoding="utf-8"))
    _, records = assert_manifest_matches(DEFAULT_ASSIGNMENTS, DEFAULT_DATASET)
    dataset = replay_training_loader(DEFAULT_DATASET, trainer)
    if len(dataset) != len(records) or any(r["row_index"] != i for i, r in enumerate(records)):
        raise ValueError("training loader and assignment alignment differ")
    gold = load_jsonl(DEFAULT_GOLD_EVAL)
    report, selected, blocked = audit_selection(dataset, records, load_jsonl(DEFAULT_GOLD_DISPOSITIONS), gold)
    persisted, index = read_index(DEFAULT_PERSIST_DIR / "chroma.sqlite3")
    parity = compare_index(selected, persisted, blocked, gold)
    state = json.loads(TRAINER_STATE.read_text(encoding="utf-8"))
    exposure = training_exposure(dataset, records, blocked, gold, training_manifest, state,
                                 read_training_log(TRAIN_RUN / "raw/train.log"),
                                 saved_training_arguments(TRAINER_STATE.parent / "training_args.bin"))
    candidates, rejected = development_questions(dataset, records, selected, blocked, gold,
                                                seeds=QUESTION_SEEDS + MORE_QUESTIONS)
    if rejected:
        raise ValueError(f"candidate review requires explicit revision, rejected={rejected}")
    reviewed = review_candidates(candidates, dataset)
    from host_finetune.split_groups import near_duplicate_pairs, normalize_text
    near_questions = near_duplicate_pairs([normalize_text(c["question"]) for c in reviewed])
    near_sources = near_duplicate_pairs([normalize_text(dataset[c["source_row_id"]]["output"]) for c in reviewed])
    train_commit = training_manifest["git"]["commit"]
    code_parity = {}
    for relative in ("host_finetune/finetune.py", "host_finetune/config.py", "host_finetune/llama_chat_format.py"):
        historical = subprocess.check_output(["git", "show", f"{train_commit}:{relative}"], cwd=ROOT)
        current = subprocess.check_output(["git", "show", f"HEAD:{relative}"], cwd=ROOT)
        code_parity[relative] = {"historical_blob_sha256": hashlib.sha256(historical).hexdigest(),
                                 "current_commit_blob_sha256": hashlib.sha256(current).hexdigest(),
                                 "same_at_recorded_commit_and_HEAD": historical == current,
                                 "worktree_equal_after_newline_folding": (ROOT / relative).read_text(encoding="utf-8") == current.decode().replace("\r\n", "\n")}
    checks = {
        "training_dataset_hash_matches_run": file_sha256(DEFAULT_DATASET) == training_manifest["data"]["processed_dataset_hash"],
        "split_hash_matches_run": file_sha256(DEFAULT_ASSIGNMENTS) == training_manifest["data"]["split_manifest_hash"],
        "training_loader_code_matches_recorded_commit": all(v["same_at_recorded_commit_and_HEAD"] and v["worktree_equal_after_newline_folding"] for v in code_parity.values()),
        "completed_two_epoch_run_evidence": exposure["completion_evidence_consistent"],
        "gold_family_counts_reproduced": report["overlap_rows_by_split"] == {"train": 184, "validation": 16, "test": 4},
        "index_parity": parity["document_count"] == 21193 and not any(parity[k] for k in ("missing_ids", "unexpected_ids", "content_or_metadata_mismatch_ids", "blocked_family_ids", "non_train_ids", "remaining_lexical_overlaps")) and not parity["train_26148_present"],
        "24_distinct_provenanced_candidates": len(reviewed) == 24 and len({c["group_id"] for c in reviewed}) == 24 and all(c["evidence_available"] and c["source_body_contains_output_normalized"] for c in reviewed),
        "no_candidate_lexical_duplicates": not near_questions and not near_sources and len({normalize_text(c["question"]) for c in reviewed}) == len(reviewed) and len({normalize_text(dataset[c["source_row_id"]]["output"]) for c in reviewed}) == len(reviewed),
        "snapshotted_inputs_and_prior_runs_unchanged": all(file_sha256(ROOT / p) == h for p, h in hashes.items()),
    }
    for name in ("raw", "metrics", "config", "figures"):
        (out / name).mkdir(parents=True, exist_ok=False)
    write_json(out / "raw/training_exposure.json", exposure)
    write_json(out / "raw/retrieval_isolation.json", {"selection": report, "persisted_index": parity, "index_identity": index})
    write_json(out / "raw/provenance.json", {"timestamp": timestamp, "git": revision, "input_sha256": hashes, "training_code_commit_parity": code_parity})
    (out / "raw/development_questions.jsonl").write_text("".join(json.dumps(c) + "\n" for c in reviewed), encoding="utf-8")
    write_json(out / "raw/question_review.json", {"initial_count_explanation": "Only 12 seeds were supplied; all 12 passed. This was a preparation omission, not an eligibility limit. No target of 24 was encoded in tasks.json or the original code.",
                                               "count": len(reviewed), "topic_areas": dict(Counter(c["topic_area"] for c in reviewed)),
                                               "rejected": rejected, "near_duplicate_question_pairs": near_questions, "near_duplicate_source_pairs": near_sources,
                                               "limitations": ["All sources are train and retrieval members, so questions are not independent final tests.", "Family and lexical separation are verified; broad themes remain shared with gold (e.g. founder departure and enterprise sales).", "Blog sources emphasize founder advice; they do not represent all media, claim types, or the author's corpus.", "Source presence and manual relevance review do not prove retrieval recall or validate a future ontology.", "Source-era numerical assertions require dates, attribution, and source spans in future work. No factual claims were newly scored."]})
    write_json(out / "metrics/checks.json", {"checks": checks, "failures": [k for k, v in checks.items() if not v]})
    write_json(out / "config/audit.json", {"command": "python -m host_finetune.gold_partition_audit --review", "question_seeds": QUESTION_SEEDS + MORE_QUESTIONS,
                                          "selection_rules": "unchanged: train, blog, unchanged role, selected non-gold family, one per family, no existing gold lexical hit"})
    (out / "config/packages.txt").write_text("\n".join(sorted(f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions())) + "\n", encoding="utf-8")
    write_yaml(out / "manifest.yaml", {
        "experiment_id": REVIEW_ID, "timestamp": timestamp, "status": "MEASURED" if all(checks.values()) else "incomplete",
        "execution_commands": [subprocess.list2cmdline([sys.executable, "-m", "host_finetune.gold_partition_audit", "--review"])], "git": revision,
        "data": {"source_manifest": "raw/provenance.json", "source_hashes": hashes, "processed_dataset_hash": file_sha256(DEFAULT_DATASET), "split_manifest_hash": file_sha256(DEFAULT_ASSIGNMENTS), "gold_eval_hash": file_sha256(DEFAULT_GOLD_EVAL)},
        "rag": {"persona_profile_hash": "n/a: no retrieval executed", "chroma_collection": TRAIN_ONLY_COLLECTION, "index_version_or_snapshot": index, "embedding_model": "nomic-embed-text (recorded; not loaded)"},
        "model": {"model_name": "n/a: historical Relabel input audit, no model executed", "model_or_adapter_hash": "n/a: weights not accessed", "config": "config/audit.json", "random_seed": "n/a: deterministic audit; historical train seed 42"},
        "evaluation": {"evaluator_commit": revision["commit"], "judge_model": "n/a: no scoring", "judge_config": "n/a: no judge"},
        "environment": {"python_version": sys.version, "package_snapshot": "config/packages.txt", "hardware": f"CPU only; {platform.platform()}; {platform.machine()}; GPU unused"},
        "outputs": {"raw": ["raw/training_exposure.json", "raw/retrieval_isolation.json", "raw/provenance.json", "raw/development_questions.jsonl", "raw/question_review.json"], "metrics": ["metrics/checks.json"], "warnings_and_failures": {"audit_failures": [k for k, v in checks.items() if not v], "limitations": exposure["limitations"], "coordination": "Pending owner-controlled operational state reconciliation; see notes.md and raw/verification.json after final checks"}},
    })
    print(json.dumps({"experiment": REVIEW_ID, "checks": checks, "exposure_counts": exposure["counts"], "candidates": len(reviewed)}, indent=2))
    return int(not all(checks.values()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", action="store_true", help="new read-only training exposure and 24-question review")
    parser.add_argument("--coordination-proposal", action="store_true", help="write owner-review state copies only; do not apply them")
    args = parser.parse_args()
    if args.coordination_proposal:
        proposal = prepare_coordination_proposal(ROOT / "experiments" / REVIEW_ID)
        print(json.dumps(proposal, indent=2))
        return int(not proposal["proposed_state_full_checker_dry_run"]["ok"])
    if args.review:
        return run_review()
    # Refuse overwrite even after a failed run. No existing experiment is edited.
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    db = DEFAULT_PERSIST_DIR / "chroma.sqlite3"
    inputs = [DEFAULT_DATASET, DEFAULT_ASSIGNMENTS, DEFAULT_GOLD_DISPOSITIONS, DEFAULT_GOLD_EVAL,
              CORPUS, ROOT / "experiments/EXP-20261003-004-relabel-continuations/manifest.yaml",
              Path(__file__), ROOT / "tests/test_gold_partition_audit.py", db]
    inputs += sorted((ROOT / "data").glob("*.jsonl"))
    inputs += sorted((ROOT / "data/cleaned").glob("*.jsonl"))
    inputs += [p for suffix in ("-wal", "-shm") if (p := Path(str(db) + suffix)).exists()]
    hashes = {p.relative_to(ROOT).as_posix(): file_sha256(p) for p in inputs}
    timestamp = datetime.now(timezone.utc).isoformat()
    revision = {"commit": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"),
                "dirty": bool(git("status", "--porcelain"))}
    dataset, assignments, dispositions, gold = map(load_jsonl, (DEFAULT_DATASET, DEFAULT_ASSIGNMENTS, DEFAULT_GOLD_DISPOSITIONS, DEFAULT_GOLD_EVAL))
    report, selected, blocked = audit_selection(dataset, assignments, dispositions, gold)
    persisted, index_identity = read_index(db)
    comparison = compare_index(selected, persisted, blocked, gold)
    candidates, rejected = development_questions(dataset, assignments, selected, blocked, gold)
    published = json.loads(CORPUS.read_text(encoding="utf-8"))
    checks = {
        "gold_identity": hashes[DEFAULT_GOLD_EVAL.relative_to(ROOT).as_posix()] == "d7c4f997899e28f28eb3ea9aa3417857da6f0666acbf72fd8ac31422034e520e" and {r["id"] for r in gold} == {f"gold_{i:03d}" for i in range(1, 31)} and len(gold) == 30,
        "recorded_counts": report["family_only_selection"]["train_rows"] == 21377 and report["family_only_selection"]["excluded_gold"] == 183 and report["family_only_selection"]["indexed"] == 21194 and report["final_selection"]["excluded_gold"] == 184,
        "published_index_count": comparison["document_count"] == published["document_count"] == len(selected) == 21193,
        "held_out_absent": comparison["train_26148_present"] is published["train_26148_present"] is False,
        "membership_and_contents": not any(comparison[k] for k in ("missing_ids", "unexpected_ids", "content_or_metadata_mismatch_ids", "non_train_ids", "blocked_family_ids", "remaining_lexical_overlaps")),
        "cosine_index": index_identity["space"] == "cosine",
        "development_candidates": bool(candidates) and not rejected,
        "input_hashes_unchanged": all(file_sha256(ROOT / name) == value for name, value in hashes.items()),
    }
    limitations = [
        "Lexical audit uses existing exact/title/8-word containment and token Jaccard >=0.80 rules; semantic paraphrases and unrecorded source-family links can remain undetected.",
        "Gold is the reused 30-topic comparison benchmark, not an untouched final test.",
        "Family exclusions apply to retrieval selection, not removal from the frozen fine-tuning dataset. Overlap rows by split remain present there; no split changes are authorized.",
        "Questions are manually paraphrased proposals anchored to unchanged train blog rows. They are neither independent test items nor verified answerable benchmarks; human review is still needed.",
        "No source bodies, answers, or expected facts exported. No training, model generation, scoring, embedding, vector search, or graph access performed.",
        "SQLite metadata/content parity does not validate HNSW vectors or retrieval quality. No historical experiment was rerun.",
        "Shared project state intentionally unchanged under user instructions; task status and branch fields require owner reconciliation.",
    ]
    failures = [name for name, ok in checks.items() if not ok]
    for folder in ("raw", "metrics", "config", "figures"):
        (OUT / folder).mkdir(parents=True, exist_ok=False)
    write_json(OUT / "raw/audit.json", {**report, "persisted_index": comparison, "rejected_candidates": rejected})
    (OUT / "raw/development_questions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in candidates), encoding="utf-8")
    write_json(OUT / "raw/provenance.json", {"timestamp": timestamp, "git": revision, "input_sha256": hashes, "index": index_identity})
    write_json(OUT / "metrics/checks.json", {"checks": checks, "failures": failures, "candidate_count": len(candidates)})
    write_json(OUT / "config/audit.json", {"overlap_policy": "existing train_only_index + split_groups; unchanged threshold 0.80", "question_seeds": QUESTION_SEEDS, "read_only_sqlite": True})
    packages = sorted(f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions())
    (OUT / "config/packages.txt").write_text("\n".join(packages) + "\n", encoding="utf-8")
    (OUT / "notes.md").write_text("# TASK-002 gold-partition audit\n\n" + "\n\n".join(limitations) + "\n\nFailures: " + (", ".join(failures) or "none") + "\n", encoding="utf-8")
    # YAML emitted as conventional nested mappings to support project_pm's small parser.
    manifest = {
        "experiment_id": EXPERIMENT_ID, "timestamp": timestamp, "status": "MEASURED" if not failures else "incomplete",
        "execution_commands": [subprocess.list2cmdline([sys.executable, "-m", "host_finetune.gold_partition_audit"])],
        "git": revision,
        "data": {"source_manifest": "raw/provenance.json", "source_hashes": hashes,
                 "processed_dataset_hash": file_sha256(DEFAULT_DATASET), "split_manifest_hash": file_sha256(DEFAULT_ASSIGNMENTS), "gold_eval_hash": file_sha256(DEFAULT_GOLD_EVAL)},
        "rag": {"persona_profile_hash": "n/a: no retrieval", "chroma_collection": TRAIN_ONLY_COLLECTION,
                "index_version_or_snapshot": {**index_identity, "sqlite_sha256": file_sha256(db), "logical_content_sha256": comparison["logical_content_sha256"]},
                "embedding_model": "nomic-embed-text (recorded index model; not loaded or independently verified)"},
        "model": {"model_name": "n/a: no model executed", "model_or_adapter_hash": "n/a: no weights accessed", "config": "config/audit.json", "random_seed": "n/a: deterministic, no sampling"},
        "evaluation": {"evaluator_commit": revision["commit"], "judge_model": "n/a: no scoring", "judge_config": "n/a: no judge", "audit_code_sha256": file_sha256(Path(__file__))},
        "environment": {"python_version": sys.version, "package_snapshot": "config/packages.txt", "hardware": f"CPU only; {platform.platform()}; {platform.machine()}; GPU unused"},
        "outputs": {"raw": ["raw/audit.json", "raw/provenance.json", "raw/development_questions.jsonl"], "metrics": ["metrics/checks.json"], "warnings_and_failures": {"failures": failures, "limitations": limitations}},
    }
    def yaml_lines(obj: dict, indent: int = 0) -> list[str]:
        result = []
        for key, value in obj.items():
            if isinstance(value, dict):
                result.append(" " * indent + key + ":")
                result.extend(yaml_lines(value, indent + 2))
            else:
                result.append(" " * indent + key + ": " + json.dumps(value))
        return result
    (OUT / "manifest.yaml").write_text("\n".join(yaml_lines(manifest)) + "\n", encoding="utf-8")
    print(json.dumps({"experiment": EXPERIMENT_ID, "checks": checks, "candidates": len(candidates), "failures": failures}, indent=2))
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
