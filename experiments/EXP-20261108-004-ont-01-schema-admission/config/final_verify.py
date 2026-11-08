"""Post-build contract verification; preserves initial run outputs and hashes."""
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from host_finetune.gold_partition_audit import write_json
from host_finetune.ontology_contract import fingerprint, snapshot_from_dict, validate_snapshot
from host_finetune.split_groups import file_sha256
from host_finetune.train_only_index import load_jsonl

RUN = ROOT / "experiments/EXP-20261108-004-ont-01-schema-admission"


def main():
    if (RUN / "raw/verification.json").exists():
        raise FileExistsError("refusing to overwrite verification evidence")
    pilot = json.loads((RUN / "raw/pilot_admission.json").read_text(encoding="utf-8"))
    snapshot = snapshot_from_dict(pilot["snapshot"])
    source = ROOT / "data/cleaned/jasonlemkin_blog.jsonl"
    records = load_jsonl(source)
    bodies = {d.id: records[d.provenance.source_line - 1]["content"] for d in snapshot.documents}
    for doc in snapshot.documents:
        p = doc.provenance
        assert p.cleaned_file_sha256 == file_sha256(source)
        assert p.raw_file_sha256 == file_sha256(ROOT / "data" / p.source_file)
        assert p.cleaned_record_sha256 == fingerprint(records[p.source_line - 1])
        assert p.dataset_sha256 == file_sha256(ROOT / "experiments/EXP-20261003-004-relabel-continuations/raw/dataset.jsonl")
        assert p.split_sha256 == file_sha256(ROOT / "experiments/EXP-20261003-004-relabel-continuations/raw/split_assignments.jsonl")
        assert p.gold_sha256 == file_sha256(ROOT / "data/openai_ft/lemkin_gold_eval.jsonl")
        assert p.selection_code_sha256 == file_sha256(ROOT / "host_finetune/train_only_index.py")
    validation = validate_snapshot(snapshot, bodies)
    assert len(validation.documents) == 5 and all(v.state.value == "pending_review" for v in validation.documents)
    assert not validation.assertions
    initial = json.loads((RUN / "raw/provenance.json").read_text(encoding="utf-8"))["input_sha256"]
    final_hashes = {p: file_sha256(ROOT / p) for p in initial}
    changed = [p for p, digest in initial.items() if final_hashes[p] != digest]
    assert set(changed) <= {"host_finetune/ontology_contract.py", "tests/test_ontology_contract.py"}
    archive = RUN / "config/source_snapshot"
    archive.mkdir(exist_ok=False)
    for relative in ("host_finetune/ontology_contract.py", "tests/test_ontology_contract.py",
                     "host_finetune/graph_ready.py", "host_finetune/train_only_index.py",
                     "host_finetune/split_groups.py"):
        destination = archive / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / relative).read_bytes())
    write_json(RUN / "raw/final_validation.json", {
        "timestamp": datetime.now(timezone.utc).isoformat(), "validation": asdict(validation),
        "final_input_sha256": final_hashes, "post_build_code_changes": changed,
        "post_build_change_reason": "Exact Decimal monetary precision and invalid year-zero quarter regression hardening; initial recorded builder output retained and manifest revalidated with final contracts.",
        "protected_data_hashes_unchanged": True, "pending": 5, "admitted": 0, "real_assertions": 0,
    })
    modules = ["test_ontology_contract", "test_evidence_capsule", "test_numeric_grounding", "test_factual_reliability",
               "test_graph_ready", "test_retrieval_by_medium", "test_train_only_index", "test_split_groups",
               "test_gold_partition_audit", "test_project_pm"]
    commands = [
        [sys.executable, "-m", "pytest", *[f"tests/{name}.py" for name in modules], "-q"],
        [sys.executable, "-m", "project_pm", "check-manifest", str((RUN / "manifest.yaml").relative_to(ROOT))],
        [sys.executable, "-m", "project_pm", "check"],
    ]
    results = []
    for command in commands:
        run = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        results.append({"command": subprocess.list2cmdline(command), "exit_code": run.returncode,
                        "stdout": run.stdout, "stderr": run.stderr})
    write_json(RUN / "raw/verification.json", {"timestamp": datetime.now(timezone.utc).isoformat(),
                                             "contract_sha256": file_sha256(ROOT / "host_finetune/ontology_contract.py"),
                                             "tests_sha256": file_sha256(ROOT / "tests/test_ontology_contract.py"),
                                             "results": results})
    for result in results:
        print(result["stdout"])
    return int(any(r["exit_code"] for r in results))


if __name__ == "__main__":
    raise SystemExit(main())
