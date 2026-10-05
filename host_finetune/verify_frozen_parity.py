"""Read-only parity audit for the frozen Relabel file and RAG selection.

This module does not import the trainer, rebuild_chroma, or
audit_retrieval_corpus. It does not construct a Chroma client, embed text,
or write datasets. The persisted collection is read through SQLite with
``mode=ro``.

Exit status is 0 when every blocking check passes. Non-blocking rows can
be FAIL or UNVERIFIED and still leave the process at 0. Those rows are the
known reproducibility gaps and the deferred live-collection audit.

Balance policy, membership, order, and explicit-CRLF byte parity are
blocking. The historical command that wrote the balanced file stays a
non-blocking UNVERIFIED gap.
"""
from __future__ import annotations

import functools
import hashlib
import json
import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from host_finetune.queue_followup_trains import select_balanced_rows
from host_finetune.split_groups import file_sha256
from host_finetune.train_only_index import (
    DEFAULT_ASSIGNMENTS,
    DEFAULT_DATASET,
    DEFAULT_GOLD_DISPOSITIONS,
    DEFAULT_GOLD_EVAL,
    DEFAULT_PERSIST_DIR,
    PROTECTED_COLLECTION,
    TRAIN_ONLY_COLLECTION,
    ProtectedCollectionError,
    assert_collection_allowed,
    assignment_rows,
    blocked_group_ids,
    headline_overlap_groups,
    load_jsonl,
    select_train_documents,
)

ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_COLLECTION = TRAIN_ONLY_COLLECTION
INFERENCE_PARAMETERS = (
    ROOT / "experiments" / "EXP-20261005-003-relabel-rag-final" / "config" / "run_parameters.json"
)

DATASET_SHA256 = "1231835ecc5dcb608fefa0357d325affb2afbdc3aeade2a63b1dc42b2920897b"
SPLIT_SHA256 = "6e31aedf33bbc81488641e3a2d0d284199bd7c3e724d97d27e801a37010ea7fb"
BALANCED_SHA256 = "6e99b414733eafdfa27a5e94dd26beff1498bbfd58af132251e24044fef4f057"
GOLD_SHA256 = "d7c4f997899e28f28eb3ea9aa3417857da6f0666acbf72fd8ac31422034e520e"
EXP008_SPLIT_SHA256 = "1e3ae21d2dd857c097b8d6816bb0941376d3f6719d142bd0f78a8c1848a8b3f6"

EXPECTED_TOTAL = 26545
EXPECTED_SPLITS = {"train": 21377, "validation": 2588, "test": 2580}
EXPECTED_FAMILY_INDEXED = 21194
EXPECTED_FAMILY_EXCLUDED = 183
EXPECTED_RAG = 21193
EXPECTED_RAG_ROLES = {"unchanged": 11611, "opening": 1687, "continuation": 7895}
HELD_OUT_ID = "train_26148"
METADATA_FIELDS = ("group_id", "medium", "row_id", "split")

BALANCED_PATH = ROOT / "host_finetune" / "data" / "dataset_fit512_keepbreaks_balanced.jsonl"
KEEPBREAKS_PATH = (
    ROOT / "host_finetune" / "data" / "dataset_from_cleaned_sources_fit512_keepbreaks.jsonl"
)
HASH_INVENTORY = ROOT / "docs" / "data" / "artifact_hashes.csv"
EXP005_COMMANDS = ROOT / "experiments" / "EXP-20261003-005-relabel-qlora" / "config" / "commands.txt"
EXP008_SPLIT = (
    ROOT / "experiments" / "EXP-20261001-008-repaired-balanced-split" / "raw" / "split_assignments.jsonl"
)
EXP009_CORPUS = ROOT / "experiments" / "EXP-20261004-009-dense-rag-v1" / "metrics" / "corpus.json"
EXP009_RETRIEVER = ROOT / "experiments" / "EXP-20261004-009-dense-rag-v1" / "config" / "retriever.json"
EXP009_COMMANDS = ROOT / "experiments" / "EXP-20261004-009-dense-rag-v1" / "config" / "commands.txt"

IDENTITY_SHA256 = {
    "data/jasonlemkin_blog.jsonl": "8492ef11acacd767ce6bcdbdf6215467b6ba74d3e57dc50ffcd0af7deebc34a8",
    "data/jasonlemkinlinkedin.jsonl": "a2899576ac75ebb5fe05af15bdd0425eef9636de42bb058e82b9c456ae71ef09",
    "data/jasonlk_originals.jsonl": "a2021528bf4a34cd34c074d252d6bf246fddbe422033e7b88ebc0e11c74044c1",
    "data/jasonmlemkinyoutubetranscripts.jsonl": "a943eb049b79b75e6aa17073cfcd4a7431e7a163b3a4276f4f8116cc31a1e337",
    "data/saastryoutubetranscripts.jsonl": "89e4de78a7e3e947625523b27fbc29a0b75ef217f5c6e7a76d1aaca41916560d",
    "data/cleaned/jasonlemkin_blog.jsonl": "07dc0623f567ec5e05cee71b3577e62be143ddf651bec4257b9955593f2a7bab",
    "data/cleaned/jasonlemkinlinkedin.jsonl": "da4361c56f93820291fb583d3386e7a211aadaf9e66d8d0a971873a9920ccbf2",
    "data/cleaned/jasonlk_originals.jsonl": "e779f42fbb951d6ea4da9e8e35862563bd267d0fb635c2619aefb1ba593770c5",
    "data/cleaned/jasonmlemkinyoutubetranscripts.jsonl": "e13d7f0a45b4279281dd2ae4946a702b9036b4349ea90a220364d017a18e8484",
    "data/cleaned/saastryoutubetranscripts.jsonl": "b673ee254840edd2e26bfcd3f8ab88d5033ba8b9f896de73eaccafbf21024cb7",
    "data/openai_ft/lemkin_gold_eval.jsonl": GOLD_SHA256,
}

DIRTY_WORKTREES = (
    "EXP-20261001-008 commit ebe5d8ebb4a9a8b5d700b45c7d50d40930c55a36 dirty",
    "EXP-20261003-004 commit ea1d58b30cafcb4cf18a03ded39963f64ce6aee7 dirty",
    "EXP-20261003-005 commit ea1d58b30cafcb4cf18a03ded39963f64ce6aee7 dirty",
    "EXP-20261004-009 commit 4006d4cee7768789c460c4c2ae1b1fcb1b124b03 dirty",
)


@dataclass(frozen=True)
class Check:
    """One parity row. ``blocking`` failures make the process exit 1."""

    name: str
    expected: str
    actual: str
    status: str
    evidence: str
    blocking: bool = True

    def detail(self) -> str:
        gate = "blocking" if self.blocking else "gap"
        return (
            f"{self.name} [{self.status}, {gate}]\n"
            f"  expected: {self.expected}\n"
            f"  actual: {self.actual}\n"
            f"  evidence: {self.evidence}"
        )


def _status(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def _read_inventory(path: Path = HASH_INVENTORY) -> dict[str, str]:
    rows: dict[str, str] = {}
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        if not line.strip():
            continue
        rel, _bytes, digest = line.split(",", 2)
        rows[rel] = digest.strip()
    return rows


def _load_pair() -> tuple[list[dict], list[dict]]:
    return load_jsonl(DEFAULT_DATASET), load_jsonl(DEFAULT_ASSIGNMENTS)


@functools.cache
def _selection() -> dict:
    """In-memory RAG replay. No embedding and no Chroma client."""
    dataset, assignments = _load_pair()
    dispositions = load_jsonl(DEFAULT_GOLD_DISPOSITIONS)
    gold = load_jsonl(DEFAULT_GOLD_EVAL)
    family_groups = blocked_group_ids(assignments, dispositions)
    family_docs, family_stats = select_train_documents(dataset, assignments, family_groups)
    extra_groups, extra_rows = headline_overlap_groups(
        dataset, assignments, gold, family_groups
    )
    rag_docs, rag_stats = select_train_documents(
        dataset, assignments, family_groups | extra_groups
    )
    family_ids = {doc["id"] for doc in family_docs}
    rag_ids = {doc["id"] for doc in rag_docs}
    return {
        "dataset": dataset,
        "assignments": assignments,
        "family_docs": family_docs,
        "family_stats": family_stats,
        "extra_rows": extra_rows,
        "rag_docs": rag_docs,
        "rag_stats": rag_stats,
        "removed_ids": family_ids - rag_ids,
    }


def _records() -> list[dict]:
    if not DEFAULT_ASSIGNMENTS.is_file():
        return []
    return assignment_rows(load_jsonl(DEFAULT_ASSIGNMENTS))


def _hash_check(name: str, path: Path, expected: str, evidence: str) -> Check:
    if not path.is_file():
        return Check(name, expected, "missing", "FAIL", evidence)
    actual = file_sha256(path)
    return Check(name, expected, actual, _status(actual == expected), evidence)


def _relabel_count_check() -> Check:
    evidence = "function=assignment_rows on EXP-004 split_assignments.jsonl"
    if not DEFAULT_DATASET.is_file() or not DEFAULT_ASSIGNMENTS.is_file():
        return Check(
            "relabel_counts",
            str(EXPECTED_SPLITS),
            "missing dataset or assignments",
            "FAIL",
            evidence,
        )
    dataset, assignments = _load_pair()
    records = assignment_rows(assignments)
    counts = Counter(row.get("split") for row in records)
    actual = {
        "total": len(records),
        "dataset_rows": len(dataset),
        **{name: counts.get(name, 0) for name in EXPECTED_SPLITS},
    }
    ok = (
        len(records) == EXPECTED_TOTAL
        and len(dataset) == EXPECTED_TOTAL
        and all(counts.get(name) == EXPECTED_SPLITS[name] for name in EXPECTED_SPLITS)
    )
    expected = f"total={EXPECTED_TOTAL} {EXPECTED_SPLITS}"
    return Check("relabel_counts", expected, str(actual), _status(ok), evidence)


def _group_check() -> Check:
    evidence = "function=assignment group_id; split ownership, not RAG overlap"
    records = _records()
    splits_of: dict[str, set[str]] = {}
    for row in records:
        splits_of.setdefault(row.get("group_id"), set()).add(row.get("split"))
    crossed = sorted(group for group, splits in splits_of.items() if len(splits) > 1)
    actual = f"groups={len(splits_of)} crossed={len(crossed)}"
    if crossed:
        actual += " example=" + crossed[0]
    return Check(
        "group_single_split",
        "every group_id is in one split",
        actual,
        _status(not crossed and len(splits_of) > 0),
        evidence,
    )


def _sft_command_check() -> Check:
    evidence = "file=EXP-20261003-005 config/commands.txt; finetune.py was not imported"
    expected = (
        "FINETUNE_DATASET EXP-004 dataset.jsonl; "
        "SPLIT_MANIFEST EXP-004 split_assignments.jsonl; "
        "ADAPTER_DIR lemkin_lora_relabel"
    )
    if not EXP005_COMMANDS.is_file():
        return Check("sft_command_source", expected, "missing commands.txt", "FAIL", evidence)
    text = EXP005_COMMANDS.read_text(encoding="utf-8").replace("\\", "/")
    ok = (
        "experiments/EXP-20261003-004-relabel-continuations/raw/dataset.jsonl" in text
        and "experiments/EXP-20261003-004-relabel-continuations/raw/split_assignments.jsonl" in text
        and "lemkin_lora_relabel" in text
    )
    return Check("sft_command_source", expected, "command text matched" if ok else "command text mismatch", _status(ok), evidence)


def _rag_checks(bundle: dict | None, error: str | None) -> list[Check]:
    evidence = (
        "functions=blocked_group_ids, headline_overlap_groups, select_train_documents; "
        "in memory; no embed; no Chroma write"
    )
    if error or bundle is None:
        actual = error or "selection did not run"
        names = (
            "rag_family_selection",
            "rag_final_count",
            "rag_held_out_absent",
            "rag_exclusion_split",
            "rag_all_train",
            "rag_text_is_output",
            "rag_roles",
            "rag_metadata",
        )
        return [Check(name, "see parity contract", actual, "FAIL", evidence) for name in names]

    family_stats = bundle["family_stats"]
    rag_stats = bundle["rag_stats"]
    removed = bundle["removed_ids"]
    rag_docs = bundle["rag_docs"]
    dataset = bundle["dataset"]
    records = assignment_rows(bundle["assignments"])
    by_index = {row["row_index"]: row for row in records}

    family_ok = (
        family_stats.get("train_rows") == EXPECTED_SPLITS["train"]
        and family_stats.get("excluded_gold") == EXPECTED_FAMILY_EXCLUDED
        and family_stats.get("indexed") == EXPECTED_FAMILY_INDEXED
        and family_stats.get("empty_output") == 0
    )
    family = Check(
        "rag_family_selection",
        f"train={EXPECTED_SPLITS['train']} excluded_gold={EXPECTED_FAMILY_EXCLUDED} indexed={EXPECTED_FAMILY_INDEXED}",
        str(family_stats),
        _status(family_ok),
        evidence + "; first call uses gold_overlap_family only",
    )
    final_ok = rag_stats.get("indexed") == EXPECTED_RAG and len(rag_docs) == EXPECTED_RAG
    final = Check(
        "rag_final_count",
        str(EXPECTED_RAG),
        str(rag_stats.get("indexed")),
        _status(final_ok),
        evidence + "; second call adds headline overlap",
    )
    held = Check(
        "rag_held_out_absent",
        f"{HELD_OUT_ID} absent",
        f"present={HELD_OUT_ID in {doc['id'] for doc in rag_docs}} removed={sorted(removed)}",
        _status(HELD_OUT_ID not in {doc["id"] for doc in rag_docs}),
        evidence,
    )
    exclusion_ok = removed == {HELD_OUT_ID} and (
        rag_stats.get("excluded_gold") == family_stats.get("excluded_gold", -1) + 1
    )
    exclusion = Check(
        "rag_exclusion_split",
        f"removed ids={{{HELD_OUT_ID}}} and excluded_gold increases by 1 from the family call",
        f"removed={sorted(removed)} family_excluded={family_stats.get('excluded_gold')} rag_excluded={rag_stats.get('excluded_gold')}",
        _status(exclusion_ok),
        evidence,
    )
    bad_split = [
        doc["id"] for doc in rag_docs if doc["metadata"].get("split") != "train"
    ]
    split_check = Check(
        "rag_all_train",
        "every selected row has split=train",
        f"non_train={len(bad_split)}",
        _status(not bad_split and bool(rag_docs)),
        evidence,
    )

    text_mismatches = 0
    instruction_swaps = 0
    empty = 0
    for doc in rag_docs:
        row_id = doc["metadata"].get("row_id")
        row = dataset[row_id] if isinstance(row_id, int) and 0 <= row_id < len(dataset) else {}
        output = (row.get("output") or "").strip()
        instruction = (row.get("instruction") or "").strip()
        if not doc["text"]:
            empty += 1
        if doc["text"] != output:
            text_mismatches += 1
        if doc["text"] == instruction and instruction != output:
            instruction_swaps += 1
    text_check = Check(
        "rag_text_is_output",
        "text equals stripped output; instruction is not substituted; none empty",
        f"output_mismatches={text_mismatches} instruction_swaps={instruction_swaps} empty={empty}",
        _status(text_mismatches == 0 and instruction_swaps == 0 and empty == 0 and bool(rag_docs)),
        "function=select_train_documents reads output",
    )

    roles: Counter[str] = Counter()
    for doc in rag_docs:
        rec = by_index.get(doc["metadata"].get("row_id"), {})
        roles[str(rec.get("sft_role") or "")] += 1
    role_actual = {name: roles.get(name, 0) for name in EXPECTED_RAG_ROLES}
    role_check = Check(
        "rag_roles",
        str(EXPECTED_RAG_ROLES),
        str(role_actual),
        _status(role_actual == EXPECTED_RAG_ROLES),
        "sft_role from EXP-004 assignments; role is not a drop filter",
    )

    incomplete = 0
    for doc in rag_docs:
        meta = doc["metadata"]
        if any(meta.get(field) in (None, "") for field in METADATA_FIELDS):
            incomplete += 1
        elif not isinstance(meta.get("row_id"), int):
            incomplete += 1
    meta_check = Check(
        "rag_metadata",
        "group_id, medium, row_id, split present on every row",
        f"incomplete={incomplete} fields={METADATA_FIELDS}",
        _status(incomplete == 0 and bool(rag_docs)),
        evidence,
    )
    return [family, final, held, exclusion, split_check, text_check, role_check, meta_check]


def _collection_guard() -> Check:
    evidence = "function=assert_collection_allowed; production collection constant"
    try:
        assert_collection_allowed(PROTECTED_COLLECTION)
        refused = False
    except ProtectedCollectionError:
        refused = True
    ok = refused and PRODUCTION_COLLECTION == "lemkin_train_only"
    actual = f"lemkin_content_refused={refused} production={PRODUCTION_COLLECTION}"
    return Check("historical_index_guard", "refuse lemkin_content; production is lemkin_train_only", actual, _status(ok), evidence)


def _identity_check() -> Check:
    evidence = "function=file_sha256 compared with docs/data/artifact_hashes.csv and this module"
    inventory = _read_inventory()
    mismatches: list[str] = []
    for rel, expected in IDENTITY_SHA256.items():
        recorded = inventory.get(rel)
        if recorded != expected:
            mismatches.append(f"{rel} inventory={recorded}")
            continue
        path = ROOT / rel
        if not path.is_file():
            mismatches.append(f"{rel} missing")
            continue
        actual = file_sha256(path)
        if actual != expected:
            mismatches.append(f"{rel} file={actual}")
    ok = not mismatches
    actual = f"{len(IDENTITY_SHA256)} files match" if ok else "; ".join(mismatches)
    return Check("raw_cleaned_gold_hashes", "inventory and file bytes match the contract", actual, _status(ok), evidence)


def _exp008_split_distinguished() -> Check:
    evidence = (
        "EXP-004 split hash is the relabel assignment. "
        "EXP-008 hash is a different file and is not the Relabel split hash."
    )
    expected = f"EXP-008 file hash {EXP008_SPLIT_SHA256} and it differs from {SPLIT_SHA256}"
    if not EXP008_SPLIT.is_file():
        return Check("exp008_split_not_relabel_split", expected, "missing EXP-008 assignments", "FAIL", evidence)
    actual_hash = file_sha256(EXP008_SPLIT)
    ok = actual_hash == EXP008_SPLIT_SHA256 and actual_hash != SPLIT_SHA256
    return Check("exp008_split_not_relabel_split", expected, actual_hash, _status(ok), evidence)


def _balance_file_check() -> Check:
    return _hash_check(
        "balanced_file_hash",
        BALANCED_PATH,
        BALANCED_SHA256,
        "hashed in place; file was not rewritten; function=file_sha256",
    )


def explicit_crlf_jsonl_bytes(rows: list[dict]) -> bytes:
    """Historical balanced-file contract: UTF-8, no BOM, CRLF after every record.

    CR and LF are literal bytes. This does not open a file and does not follow
    the host default newline.
    """
    encoded = [json.dumps(row, ensure_ascii=False).encode("utf-8") for row in rows]
    if not encoded:
        return b""
    return b"\r\n".join(encoded) + b"\r\n"


def _row_identity(row: dict) -> tuple:
    return (
        row.get("source"),
        row.get("source_file"),
        row.get("source_line"),
        row.get("instruction"),
        row.get("output"),
    )


def _balance_checks() -> list[Check]:
    evidence = (
        "function=select_balanced_rows on dataset_from_cleaned_sources_fit512_keepbreaks.jsonl, "
        "in memory. Artifact bytes are json.dumps(ensure_ascii=False) joined by explicit "
        "b'\\r\\n' plus a final CRLF. No file was written. Text mode was not used."
    )
    invocation = Check(
        "balance_invocation",
        "a recorded command that wrote dataset_fit512_keepbreaks_balanced.jsonl",
        "UNVERIFIED historical command is unrecorded; the transformation is a separate PASS",
        "UNVERIFIED",
        "TRANSFORMATION = reproducible. INVOCATION / historical command = unrecorded. "
        "build_balanced reads a different parent and writes dataset_fit512_no_saastr_xcap.jsonl.",
        blocking=False,
    )
    names = ("balance_policy", "balance_membership", "balance_order", "balance_byte_parity")
    if not KEEPBREAKS_PATH.is_file() or not BALANCED_PATH.is_file():
        missing = "parent or frozen balanced file missing"
        return [Check(name, "present files", missing, "FAIL", evidence) for name in names] + [
            invocation
        ]

    parent = load_jsonl(KEEPBREAKS_PATH)
    frozen = load_jsonl(BALANCED_PATH)
    kept, stats = select_balanced_rows(parent)
    sources = Counter(row.get("source") for row in kept)
    policy_ok = (
        stats.get("rows_in") == 43012
        and stats.get("rows_out") == 31328
        and stats.get("blog_rows") == 14315
        and stats.get("dropped_youtube_saastr") == 5492
        and sources.get("blog", 0) == 14315
        and sources.get("x", 0) == 14316
        and sources.get("youtube_saastr", 0) == 0
    )
    policy = Check(
        "balance_policy",
        "rows_in=43012 rows_out=31328 blog=14315 x=14316 youtube_saastr=0 dropped_youtube_saastr=5492",
        (
            f"rows_in={stats.get('rows_in')} rows_out={stats.get('rows_out')} "
            f"blog={sources.get('blog', 0)} x={sources.get('x', 0)} "
            f"youtube_saastr={sources.get('youtube_saastr', 0)} "
            f"dropped_youtube_saastr={stats.get('dropped_youtube_saastr')}"
        ),
        _status(policy_ok),
        evidence,
    )
    kept_ids = [_row_identity(row) for row in kept]
    frozen_ids = [_row_identity(row) for row in frozen]
    membership_ok = Counter(kept_ids) == Counter(frozen_ids)
    membership = Check(
        "balance_membership",
        "replay identity multiset equals the frozen file",
        f"replay={len(kept_ids)} frozen={len(frozen_ids)} equal={membership_ok}",
        _status(membership_ok),
        evidence + " identity=source, source_file, source_line, instruction, output",
    )
    order_ok = kept == frozen and all(
        list(left.keys()) == list(right.keys()) for left, right in zip(kept, frozen)
    )
    order = Check(
        "balance_order",
        "same parsed objects in the same order, including JSON key order",
        f"rows={len(kept)} parsed_and_key_order_equal={order_ok}",
        _status(order_ok),
        evidence,
    )
    artifact = explicit_crlf_jsonl_bytes(kept)
    replay = hashlib.sha256(artifact).hexdigest()
    on_disk = BALANCED_PATH.read_bytes()
    byte_ok = (
        replay == BALANCED_SHA256
        and artifact == on_disk
        and not artifact.startswith(b"\xef\xbb\xbf")
    )
    byte_parity = Check(
        "balance_byte_parity",
        f"sha256 {BALANCED_SHA256} via explicit CRLF bytes",
        f"sha256 {replay} equal_to_frozen_bytes={artifact == on_disk}",
        _status(byte_ok),
        evidence + " LEVEL 1 transformation. Historical invocation remains balance_invocation.",
    )
    return [policy, membership, order, byte_parity, invocation]


def _recorded_corpus_check() -> Check:
    evidence = (
        "files=EXP-20261004-009 corpus.json and retriever.json. "
        "This is recorded evidence. The live collection was not opened."
    )
    expected = "document_count=21193 train_26148_present=false collection=lemkin_train_only space=cosine"
    if not EXP009_CORPUS.is_file() or not EXP009_RETRIEVER.is_file():
        return Check("exp009_recorded_corpus", expected, "missing EXP-009 json", "FAIL", evidence)
    corpus = json.loads(EXP009_CORPUS.read_text(encoding="utf-8"))
    retriever = json.loads(EXP009_RETRIEVER.read_text(encoding="utf-8"))
    ok = (
        corpus.get("document_count") == EXPECTED_RAG
        and corpus.get("train_26148_present") is False
        and corpus.get("splits", {}).get("train") == EXPECTED_RAG
        and retriever.get("collection") == PRODUCTION_COLLECTION
        and retriever.get("document_count") == EXPECTED_RAG
        and retriever.get("distance_metric") == "cosine"
    )
    actual = (
        f"corpus_count={corpus.get('document_count')} "
        f"train_26148_present={corpus.get('train_26148_present')} "
        f"collection={retriever.get('collection')} distance={retriever.get('distance_metric')}"
    )
    return Check("exp009_recorded_corpus", expected, actual, _status(ok), evidence)


def _sqlite_ro(path: Path) -> sqlite3.Connection:
    uri = "file:" + path.resolve().as_posix() + "?mode=ro"
    return sqlite3.connect(uri, uri=True)


def _persisted_chroma_check() -> Check:
    """Compare the on-disk collection to the in-memory 21,193-row selection."""
    expected = (
        f"collection={TRAIN_ONLY_COLLECTION} count={EXPECTED_RAG} "
        f"space=cosine train_26148 absent roles={EXPECTED_RAG_ROLES}"
    )
    db_path = DEFAULT_PERSIST_DIR / "chroma.sqlite3"
    evidence = (
        f"sqlite mode=ro on {db_path}; no PersistentClient; no upsert; "
        "ids compared as a set against select_train_documents"
    )
    if not db_path.is_file():
        return Check("persisted_chroma", expected, "missing chroma.sqlite3", "FAIL", evidence)
    selection = _selection()
    expected_docs = {doc["id"]: doc for doc in selection["rag_docs"]}
    roles_by_index = {
        int(rec["row_index"]): str(rec.get("sft_role") or "")
        for rec in assignment_rows(selection["assignments"])
    }
    try:
        connection = _sqlite_ro(db_path)
    except sqlite3.Error as exc:
        return Check("persisted_chroma", expected, f"sqlite open failed: {exc}", "FAIL", evidence)
    try:
        collections = connection.execute("SELECT name, schema_str FROM collections").fetchall()
        rows: dict[str, dict] = {}
        query = """
            SELECT e.embedding_id, m.key, m.string_value, m.int_value
            FROM embeddings e
            JOIN embedding_metadata m ON m.id = e.id
        """
        for embedding_id, key, string_value, int_value in connection.execute(query):
            slot = rows.setdefault(embedding_id, {})
            if key == "chroma:document":
                slot["text"] = string_value or ""
            elif key == "row_id":
                slot["row_id"] = int_value
            elif key in {"group_id", "medium", "split"}:
                slot[key] = string_value
        queue_count = connection.execute("SELECT COUNT(*) FROM embeddings_queue").fetchone()[0]
    finally:
        connection.close()
    names = [name for name, _schema in collections]
    space = ""
    if len(collections) == 1 and collections[0][1]:
        schema = json.loads(collections[0][1])
        space = (
            schema.get("keys", {})
            .get("#embedding", {})
            .get("float_list", {})
            .get("vector_index", {})
            .get("config", {})
            .get("space", "")
        )
    missing = sorted(set(expected_docs) - set(rows))
    unexpected = sorted(set(rows) - set(expected_docs))
    empty = 0
    missing_meta = 0
    non_train = 0
    text_mismatches = 0
    group_mismatches = 0
    medium_mismatches = 0
    split_mismatches = 0
    row_id_mismatches = 0
    required = ("group_id", "medium", "row_id", "split", "text")
    for doc_id, got in rows.items():
        if not (got.get("text") or "").strip():
            empty += 1
        if any(got.get(key) in (None, "") for key in required):
            missing_meta += 1
        if got.get("split") != "train":
            non_train += 1
        exp = expected_docs.get(doc_id)
        if exp is None:
            continue
        meta = exp["metadata"]
        if got.get("text") != exp["text"]:
            text_mismatches += 1
        if got.get("group_id") != meta["group_id"]:
            group_mismatches += 1
        if got.get("medium") != meta["medium"]:
            medium_mismatches += 1
        if got.get("split") != meta["split"]:
            split_mismatches += 1
        if got.get("row_id") != int(meta["row_id"]):
            row_id_mismatches += 1
    roles = Counter(
        roles_by_index.get(int(doc_id.split("_", 1)[1]), "") for doc_id in rows
    )
    role_actual = {name: roles.get(name, 0) for name in EXPECTED_RAG_ROLES}
    ok = (
        names == [TRAIN_ONLY_COLLECTION]
        and space == "cosine"
        and len(rows) == EXPECTED_RAG
        and len(expected_docs) == EXPECTED_RAG
        and not missing
        and not unexpected
        and "train_26148" not in rows
        and empty == 0
        and missing_meta == 0
        and non_train == 0
        and text_mismatches == 0
        and group_mismatches == 0
        and medium_mismatches == 0
        and split_mismatches == 0
        and row_id_mismatches == 0
        and role_actual == EXPECTED_RAG_ROLES
    )
    actual = (
        f"collection={names} count={len(rows)} space={space} "
        f"missing={len(missing)} unexpected={len(unexpected)} "
        f"text={text_mismatches} group_id={group_mismatches} medium={medium_mismatches} "
        f"split={split_mismatches} row_id={row_id_mismatches} "
        f"empty={empty} missing_meta={missing_meta} non_train={non_train} "
        f"train_26148={'train_26148' in rows} roles={role_actual} queue_rows={queue_count}"
    )
    return Check("persisted_chroma", expected, actual, _status(ok), evidence)


def _inference_provenance_check() -> Check:
    expected = "recorded git commit and dirty flag for EXP-20261005-003"
    evidence = f"file={INFERENCE_PARAMETERS.relative_to(ROOT)}; inference metadata was not rewritten"
    if not INFERENCE_PARAMETERS.is_file():
        return Check(
            "completed_inference_provenance",
            expected,
            "missing run_parameters.json",
            "UNVERIFIED",
            evidence,
            blocking=False,
        )
    payload = json.loads(INFERENCE_PARAMETERS.read_text(encoding="utf-8"))
    git = payload.get("git") or {}
    actual = f"commit={git.get('commit')} dirty={git.get('dirty')}; not a clean commit"
    return Check(
        "completed_inference_provenance",
        expected,
        actual,
        "UNVERIFIED",
        evidence,
        blocking=False,
    )


def _embedding_gap_check() -> Check:
    command = ""
    if EXP009_COMMANDS.is_file():
        command = " ".join(EXP009_COMMANDS.read_text(encoding="utf-8").split())
    return Check(
        "embedding_build_provenance",
        "a registered command that embedded the 21193-document collection",
        f"UNVERIFIED EXP-009 command reads the collection: {command or 'missing commands.txt'}",
        "UNVERIFIED",
        "No experiment manifest records the embed that removed train_26148. "
        "The 21194 build is described only in project notes.",
        blocking=False,
    )


def _dirty_worktree_check() -> Check:
    return Check(
        "dirty_worktrees",
        "clean git SHA for EXP-008, EXP-004, EXP-005, and EXP-009",
        "UNVERIFIED " + "; ".join(DIRTY_WORKTREES),
        "UNVERIFIED",
        "Copied from manifests. Uncommitted code was not reconstructed.",
        blocking=False,
    )


@functools.cache
def evaluate() -> tuple[Check, ...]:
    """Run every file-based check once. Safe to call from pytest more than once."""
    checks: list[Check] = [
        _relabel_count_check(),
        _hash_check(
            "relabel_dataset_hash",
            DEFAULT_DATASET,
            DATASET_SHA256,
            "file=experiments/EXP-20261003-004-relabel-continuations/raw/dataset.jsonl",
        ),
        _hash_check(
            "relabel_split_hash",
            DEFAULT_ASSIGNMENTS,
            SPLIT_SHA256,
            "file=experiments/EXP-20261003-004-relabel-continuations/raw/split_assignments.jsonl "
            "This is the post-relabel assignment, not the EXP-008 file.",
        ),
        _exp008_split_distinguished(),
        _group_check(),
        Check(
            "sft_train_count",
            str(EXPECTED_SPLITS["train"]),
            str(Counter(row.get("split") for row in _records()).get("train", 0)),
            _status(
                Counter(row.get("split") for row in _records()).get("train") == EXPECTED_SPLITS["train"]
            ),
            "function=assignment_rows; validation and test are not training rows",
        ),
        _sft_command_check(),
    ]
    try:
        bundle: dict | None = _selection()
        error = None
    except Exception as exc:  # noqa: BLE001 — parity must report the failure, not crash
        bundle = None
        error = f"{type(exc).__name__}: {exc}"
    checks.extend(_rag_checks(bundle, error))
    checks.extend(
        [
            _collection_guard(),
            _identity_check(),
            _balance_file_check(),
            *_balance_checks(),
            _recorded_corpus_check(),
            _persisted_chroma_check(),
            _embedding_gap_check(),
            _dirty_worktree_check(),
            _inference_provenance_check(),
        ]
    )
    return tuple(checks)


def blocking_failures(checks: tuple[Check, ...] | list[Check] | None = None) -> list[Check]:
    rows = evaluate() if checks is None else checks
    return [row for row in rows if row.blocking and row.status != "PASS"]


def format_report(checks: tuple[Check, ...] | list[Check] | None = None) -> str:
    rows = evaluate() if checks is None else checks
    lines = [row.detail() for row in rows]
    failed = blocking_failures(rows)
    gaps = [row for row in rows if not row.blocking and row.status != "PASS"]
    lines.append(
        f"blocking_failures={len(failed)} nonblocking_gaps={len(gaps)}"
    )
    return "\n".join(lines)


def main() -> int:
    report = format_report()
    print(report)
    return 1 if blocking_failures() else 0


if __name__ == "__main__":
    raise SystemExit(main())
