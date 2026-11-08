"""Record five pending source-admission candidates. No extraction or inference."""
from dataclasses import asdict, replace
from datetime import datetime, timezone
from importlib import metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from host_finetune.gold_partition_audit import digest_json, write_json, write_yaml
from host_finetune.graph_ready import eligible_train_ids
from host_finetune.ontology_contract import (
    AUTHOR, VERSION, AdmissionDecision, Document, Passage, Provenance, Reason,
    Review, Snapshot, admission_subject, fingerprint, sha256,
    snapshot_from_dict, validate_snapshot,
)
from host_finetune.split_groups import file_sha256
from host_finetune.train_only_index import (
    DEFAULT_ASSIGNMENTS, DEFAULT_DATASET, DEFAULT_GOLD_DISPOSITIONS, DEFAULT_GOLD_EVAL,
    assignment_rows, load_jsonl,
)

RUN_ID = "EXP-20261108-004-ont-01-schema-admission"
OUT = ROOT / "experiments" / RUN_ID
DEVELOPMENT = ROOT / "experiments/EXP-20261108-003-gold-exposure-review/raw/development_questions.jsonl"
CANDIDATES = ("dev_candidate_001", "dev_candidate_013", "dev_candidate_014", "dev_candidate_020", "dev_candidate_024")


def main():
    if (OUT / "manifest.yaml").exists() or (OUT / "raw/pilot_admission.json").exists():
        raise FileExistsError("refusing to overwrite registered run outputs")
    source = ROOT / "data/cleaned/jasonlemkin_blog.jsonl"
    raw_source = ROOT / "data/jasonlemkin_blog.jsonl"
    inputs = [DEFAULT_DATASET, DEFAULT_ASSIGNMENTS, DEFAULT_GOLD_DISPOSITIONS, DEFAULT_GOLD_EVAL,
              DEVELOPMENT, source, raw_source, Path(__file__),
              ROOT / "host_finetune/ontology_contract.py", ROOT / "tests/test_ontology_contract.py",
              ROOT / "host_finetune/graph_ready.py", ROOT / "host_finetune/train_only_index.py",
              ROOT / "host_finetune/split_groups.py", ROOT / "host_finetune/rag_active.json"]
    hashes = {p.relative_to(ROOT).as_posix(): file_sha256(p) for p in inputs}
    revision = {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
                "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())}
    dataset = load_jsonl(DEFAULT_DATASET)
    assignments = assignment_rows(load_jsonl(DEFAULT_ASSIGNMENTS))
    eligible, selection = eligible_train_ids(dataset, assignments, load_jsonl(DEFAULT_GOLD_DISPOSITIONS), load_jsonl(DEFAULT_GOLD_EVAL))
    candidates = {q["id"]: q for q in load_jsonl(DEVELOPMENT)}
    sources = load_jsonl(source)
    docs, decisions, passages, links, bodies = [], [], [], [], {}
    for candidate_id in CANDIDATES:
        candidate = candidates[candidate_id]
        assert candidate["source_file"] == source.name and candidate["medium"] == "blog"
        record = sources[candidate["source_line"] - 1]
        assert digest_json(record) == candidate["cleaned_record_sha256"]
        assert sha256(dataset[candidate["source_row_id"]]["output"]) == candidate["source_output_sha256"]
        members = [r for r in assignments if r["source_file"] == source.name and r["source_line"] == candidate["source_line"]]
        assert members and all(r["split"] == "train" and f"train_{r['row_index']}" in eligible for r in members)
        body = record["content"]  # Exact immutable cleaned string, not normalized or rewritten.
        provenance = Provenance(source.name, candidate["source_line"], file_sha256(raw_source), file_sha256(source),
                                fingerprint(record), file_sha256(DEFAULT_DATASET), file_sha256(DEFAULT_ASSIGNMENTS),
                                file_sha256(DEFAULT_GOLD_EVAL), file_sha256(ROOT / "host_finetune/train_only_index.py"),
                                tuple(r["row_index"] for r in members), tuple(sorted({r["group_id"] for r in members})),
                                selection_eligible=True)
        signals = tuple(str(record[key]).strip() for key in ("author", "byline") if record.get(key))
        doc = Document(provenance, record["url"], sha256(body), str(record.get("date")) if record.get("date") else None, signals)
        decision = AdmissionDecision(doc.id, fingerprint(doc), reasons=(Reason.PENDING_REVIEW, Reason.MISSING_ATTRIBUTION))
        decision = replace(decision, review=Review(admission_subject(doc, decision)))
        passage = Passage(doc.id, 0, len(body), sha256(body))
        docs.append(doc)
        decisions.append(decision)
        passages.append(passage)
        bodies[doc.id] = body
        links.append({"document_id": doc.id, "decision_id": decision.id, "passage_id": passage.id,
                      "development_question_id": candidate_id,
                      "cleaned_source_path": candidate["cleaned_source_path"], "body_field": "content",
                      "review_required": ["byline tied to this source version", "authorship signals", "guest/interview/third-party content",
                                          "source lineage", "quotation and endorsement boundaries"],
                      "manual_approval_recorded": False})
    snapshot = Snapshot(tuple(docs), tuple(decisions), tuple(passages))
    assert snapshot_from_dict(json.loads(json.dumps(asdict(snapshot)))) == snapshot
    validation = validate_snapshot(snapshot, bodies)
    assert len(docs) == len({d.id for d in docs}) == 5
    assert all(d.state.value == "pending_review" for d in validation.documents)
    assert not snapshot.assertions
    assert len({g for d in docs for g in d.provenance.group_ids}) == 5
    assert file_sha256(DEFAULT_GOLD_EVAL) == "d7c4f997899e28f28eb3ea9aa3417857da6f0666acbf72fd8ac31422034e520e"
    assert all(file_sha256(ROOT / p) == digest for p, digest in hashes.items())
    for folder in ("raw", "metrics", "figures"):
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    pilot = {"schema_version": "pilot_admission.v1", "contract_version": VERSION, "expected_author": AUTHOR,
             "status": "pending_human_review", "selection": "five fixed candidates from frozen development-source locators; no inferred author verification",
             "snapshot": asdict(snapshot), "source_links": links,
             "review_policy": "Do not edit existing decisions. Append a new content-bound decision referencing previous_decision_id after actual human review. Capture proof text locally and pass its hash/content to validation. No claims are eligible merely because a document is a candidate."}
    write_json(OUT / "raw/pilot_admission.json", pilot)
    write_json(OUT / "raw/provenance.json", {"timestamp": timestamp, "git": revision, "input_sha256": hashes, "selection": selection})
    write_json(OUT / "raw/validation.json", asdict(validation))
    write_json(OUT / "metrics/contracts.json", {"candidate_documents": 5, "pending_documents": 5, "admitted_documents": 0,
                                               "assertions": 0, "retrievable_assertions": 0,
                                               "input_hashes_unchanged": True, "gold_hash_preserved": True})
    (OUT / "config/packages.txt").write_text("\n".join(sorted(f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions())) + "\n", encoding="utf-8")
    write_yaml(OUT / "manifest.yaml", {
        "experiment_id": RUN_ID, "timestamp": timestamp, "status": "TESTED", "pilot_admission_status": "pending_human_review",
        "execution_commands": [subprocess.list2cmdline([sys.executable, "experiments/" + RUN_ID + "/config/build_admission.py"])],
        "git": revision,
        "data": {"source_manifest": "raw/pilot_admission.json", "source_hashes": hashes,
                 "processed_dataset_hash": file_sha256(DEFAULT_DATASET), "split_manifest_hash": file_sha256(DEFAULT_ASSIGNMENTS),
                 "gold_eval_hash": file_sha256(DEFAULT_GOLD_EVAL)},
        "rag": {"persona_profile_hash": "n/a: contract/admission only", "chroma_collection": "n/a: existing index not opened",
                "index_version_or_snapshot": "n/a: no retrieval or index build", "embedding_model": "n/a: no embeddings"},
        "model": {"model_name": "n/a: no model loaded", "model_or_adapter_hash": "n/a: no weights accessed",
                  "config": "author_claim.v1 contracts; no extraction", "random_seed": "n/a: deterministic source selection"},
        "evaluation": {"evaluator_commit": revision["commit"], "judge_model": "n/a: no scoring", "judge_config": "n/a: no judge"},
        "environment": {"python_version": sys.version, "package_snapshot": "config/packages.txt",
                        "hardware": platform.platform() + "; CPU-only; GPU unused"},
        "outputs": {"raw": ["raw/pilot_admission.json", "raw/provenance.json", "raw/validation.json"],
                    "metrics": ["metrics/contracts.json"],
                    "warnings_and_failures": ["Five candidates are not manually approved author-attributed documents; all admissions are pending.",
                                              "Zero real assertions were extracted or admitted. Trusted reviewer attestations are not authenticated by this module.",
                                              "Exact spans and typed bindings do not by themselves establish semantic truth."]},
    })
    (OUT / "notes.md").write_text("# ONT-01 schema/admission contract run\n\nFive candidate train-blog sources were selected using frozen development locators and the existing gold-family exclusion functions. Body strings were read unchanged only to validate hashes and passage offsets; no source prose was exported. All five manual admissions remain pending, with no recorded byline proof or human approval. There are zero real claims and zero retrievable assertions.\n\nThis is a deterministic contract/admission smoke test, not model evaluation, extraction, or verified author annotation. No Chroma/DuckDB/graph was opened or built and no model was loaded. Human approval is required before actual source admission; stop for ONT-01 review before ONT-02. See docs/architecture/ONT_01_REPORT.md and raw/verification.json for final tests.\n", encoding="utf-8")
    print(json.dumps({"experiment": RUN_ID, "candidate_documents": 5, "pending": 5, "admitted": 0, "real_assertions": 0}, indent=2))


if __name__ == "__main__":
    main()
