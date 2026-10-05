"""Select train-only rows for a separate Chroma collection.

The live ``lemkin_content`` collection is never a valid target. Gold-overlap
families come from the recorded complete-source dispositions and are left out
so later grounding checks do not retrieve held-out text.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from host_finetune.split_groups import (
    OVERLAP_JACCARD,
    _jaccard,
    matches_gold_field,
    normalize_overlap_text,
    tokens,
)

logger = logging.getLogger(__name__)

PROTECTED_COLLECTION = "lemkin_content"
TRAIN_ONLY_COLLECTION = "lemkin_train_only"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = (
    ROOT / "experiments" / "EXP-20261003-004-relabel-continuations" / "raw" / "dataset.jsonl"
)
DEFAULT_ASSIGNMENTS = (
    ROOT
    / "experiments"
    / "EXP-20261003-004-relabel-continuations"
    / "raw"
    / "split_assignments.jsonl"
)
DEFAULT_GOLD_DISPOSITIONS = (
    ROOT
    / "experiments"
    / "EXP-20261003-003-complete-source-sft-v3"
    / "raw"
    / "dispositions.jsonl"
)
DEFAULT_GOLD_EVAL = ROOT / "data" / "openai_ft" / "lemkin_gold_eval.jsonl"
DEFAULT_PERSIST_DIR = ROOT / "host_finetune" / "output" / "chroma_lemkin_train_only"
DEFAULT_STYLE_PATH = ROOT / "host_finetune" / "exemplars" / "fixed_style_candidates.json"
# Comparison mediums in compare_adapters.py. Talk searches the transcript
# platform stored on the train-only rows, and never another written medium.
COMPARISON_MEDIUM_TO_INDEX = {
    "blog": "blog",
    "linkedin": "linkedin",
    "x": "x",
    "talk": "youtube_jason",
}
WRITTEN_PLATFORMS = ("blog", "linkedin", "x")
STYLE_PLATFORMS = (*WRITTEN_PLATFORMS, "youtube_jason")
# Trainer memory moves a little between steps. A newly loaded embed model is
# larger than this and is treated as a GPU conflict.
EXISTING_PID_TOLERANCE_MIB = 256
NEW_PID_TOLERANCE_MIB = 32


class ProtectedCollectionError(ValueError):
    """Raised when a train-only build would write the live collection."""


def assert_collection_allowed(name: str) -> None:
    if name == PROTECTED_COLLECTION:
        raise ProtectedCollectionError(
            "refusing to delete or upsert lemkin_content"
        )


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def assignment_rows(records: list[dict]) -> list[dict]:
    return [row for row in records if row.get("record_type") != "meta"]


def gold_field_values(gold_rows: list[dict]) -> list[tuple[str, str, str]]:
    """Gold id, field name, and raw value. Empty values are omitted."""
    fields: list[tuple[str, str, str]] = []
    for row in gold_rows:
        gold_id = str(row.get("id") or "")
        for field in ("topic", "gold_reference"):
            value = row.get(field) or ""
            if str(value).strip():
                fields.append((gold_id, field, str(value)))
        for value in row.get("expected_facts") or []:
            if str(value).strip():
                fields.append((gold_id, "expected_fact", str(value)))
    return fields


def matching_gold_fields(text: str, title: str, gold_fields: list[tuple[str, str, str]]) -> list[dict]:
    """Gold fields this text hits under the overlap normalizer."""
    hits = []
    for gold_id, field, value in gold_fields:
        matched, score = matches_gold_field(text, title, value, field)
        if matched:
            hits.append({"gold_id": gold_id, "field": field, "jaccard": round(score, 4)})
    return hits


def headline_overlap_groups(
    dataset_rows: list[dict],
    assignments: list[dict],
    gold_rows: list[dict],
    already_blocked: set[str] | None = None,
) -> tuple[set[str], list[dict]]:
    """Group ids whose text matches gold after URL, dash, and punctuation folding.

    A match excludes the whole ``group_id``, including sibling chunks that do
    not themselves match. Groups already blocked by recorded dispositions are
    not listed again in the detail rows.
    """
    records = assignment_rows(assignments)
    if len(records) != len(dataset_rows):
        raise ValueError(
            f"assignment rows ({len(records)}) != dataset rows ({len(dataset_rows)})"
        )
    prior = already_blocked or set()
    prepared = []
    for gold_id, field, value in gold_field_values(gold_rows):
        value_n = normalize_overlap_text(value)
        if not value_n:
            continue
        prepared.append(
            {
                "gold_id": gold_id,
                "field": field,
                "value_n": value_n,
                "tokens": tokens(value_n),
                "words": len(value_n.split()),
            }
        )
    groups: set[str] = set()
    direct: dict[str, dict] = {}
    for index, (row, rec) in enumerate(zip(dataset_rows, records)):
        if rec.get("row_index") != index:
            raise ValueError(f"assignment row_index {rec.get('row_index')} != {index}")
        text_n = normalize_overlap_text(row.get("output") or "")
        title_n = normalize_overlap_text(rec.get("base_title") or "")
        text_tokens = tokens(text_n)
        hits = []
        for item in prepared:
            score = _jaccard(text_tokens, item["tokens"])
            matched = (
                bool(text_n) and text_n == item["value_n"]
                or (item["field"] == "topic" and bool(title_n) and title_n == item["value_n"])
                or (item["words"] >= 8 and item["value_n"] in text_n)
                or score >= OVERLAP_JACCARD
            )
            if matched:
                hits.append(
                    {"gold_id": item["gold_id"], "field": item["field"], "jaccard": round(score, 4)}
                )
        if not hits:
            continue
        group_id = rec["group_id"]
        groups.add(group_id)
        if group_id in prior or group_id in direct:
            continue
        direct[group_id] = {
            "row_id": index,
            "group_id": group_id,
            "split": rec.get("split"),
            "source_platform": rec.get("source_platform"),
            "source_file": rec.get("source_file"),
            "source_line": rec.get("source_line"),
            "hits": hits,
        }
    return groups, list(direct.values())


def blocked_group_ids(assignments: list[dict], dispositions: list[dict]) -> set[str]:
    """Families whose source document was flagged gold_overlap_family."""
    keys: set[tuple[str, int]] = set()
    groups: set[str] = set()
    for row in dispositions:
        reasons = row.get("reasons") or []
        if "gold_overlap_family" not in reasons:
            continue
        keys.add((row["source_file"], int(row["source_line"])))
        for group in row.get("prior_group_ids") or []:
            groups.add(group)
    for rec in assignment_rows(assignments):
        key = (rec.get("source_file"), int(rec["source_line"]))
        if key in keys:
            groups.add(rec["group_id"])
    return groups


def select_train_documents(
    dataset_rows: list[dict],
    assignments: list[dict],
    blocked_groups: set[str],
) -> tuple[list[dict], dict]:
    """Keep non-empty train rows whose family is not gold-overlapping."""
    records = assignment_rows(assignments)
    if len(records) != len(dataset_rows):
        raise ValueError(
            f"assignment rows ({len(records)}) != dataset rows ({len(dataset_rows)})"
        )
    docs: list[dict] = []
    stats = {
        "rows": len(dataset_rows),
        "train_rows": 0,
        "excluded_split": 0,
        "excluded_gold": 0,
        "empty_output": 0,
        "indexed": 0,
    }
    for index, (row, rec) in enumerate(zip(dataset_rows, records)):
        if rec.get("row_index") != index:
            raise ValueError(f"assignment row_index {rec.get('row_index')} != {index}")
        if rec.get("split") != "train":
            stats["excluded_split"] += 1
            continue
        stats["train_rows"] += 1
        if rec.get("group_id") in blocked_groups:
            stats["excluded_gold"] += 1
            continue
        text = (row.get("output") or "").strip()
        if not text:
            stats["empty_output"] += 1
            continue
        docs.append(
            {
                "id": f"train_{rec['row_index']}",
                "text": text,
                "metadata": {
                    "group_id": rec["group_id"],
                    "medium": rec["source_platform"],
                    "row_id": int(rec["row_index"]),
                    "split": rec["split"],
                },
            }
        )
    stats["indexed"] = len(docs)
    return docs, stats


def ollama_embed_body(texts: list[str], model: str, *, cpu: bool) -> dict:
    body: dict = {"model": model, "input": texts}
    if cpu:
        body["options"] = {"num_gpu": 0}
    return body


def vram_conflict(before: dict[int, int], after: dict[int, int]) -> str | None:
    """Return a reason when GPU memory shows a new or growing consumer."""
    for pid, used in before.items():
        later = after.get(pid)
        if later is not None and later - used > EXISTING_PID_TOLERANCE_MIB:
            return f"pid {pid} VRAM rose from {used} MiB to {later} MiB"
    for pid, used in after.items():
        if pid not in before and used > NEW_PID_TOLERANCE_MIB:
            return f"new GPU process {pid} is using {used} MiB"
    return None


def gpu_used_conflict(before: list[int], after: list[int]) -> str | None:
    """Compare total memory.used per GPU. Windows often omits per-process MiB."""
    if len(before) != len(after):
        return "GPU inventory changed during the embed probe"
    for index, (old, new) in enumerate(zip(before, after)):
        if new - old > EXISTING_PID_TOLERANCE_MIB:
            return f"GPU {index} memory rose from {old} MiB to {new} MiB"
    return None


def format_factual_excerpts(excerpts: list[str] | None) -> str:
    """Separate evidence block. Empty input yields an empty string."""
    cleaned = [text.strip() for text in (excerpts or []) if text and text.strip()]
    if not cleaned:
        return ""
    lines = [
        "Factual excerpts from train-owned source text.",
        "Use them as evidence for claims. They are not style examples.",
        "Do not copy long passages verbatim.",
    ]
    for number, text in enumerate(cleaned, 1):
        lines.append(f"[Excerpt {number}]\n{text}")
    return "\n".join(lines)


def index_documents(docs: list[dict], collection, embed_fn, batch_size: int = 64) -> int:
    """Upsert documents with a caller-supplied embedder. Returns the new count."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(docs), batch_size):
        batch = docs[start : start + batch_size]
        vectors = embed_fn([item["text"] for item in batch])
        if len(vectors) != len(batch):
            raise RuntimeError(
                f"embedder returned {len(vectors)} vectors for {len(batch)} texts"
            )
        logger.info("embedding %d / %d", start + len(batch), len(docs))
        collection.upsert(
            ids=[item["id"] for item in batch],
            documents=[item["text"] for item in batch],
            embeddings=vectors,
            metadatas=[item["metadata"] for item in batch],
        )
    return collection.count()


def index_medium_for(comparison_medium: str) -> str:
    """Map a comparison medium onto the stored ``source_platform`` value."""
    try:
        return COMPARISON_MEDIUM_TO_INDEX[comparison_medium]
    except KeyError as exc:
        known = ", ".join(sorted(COMPARISON_MEDIUM_TO_INDEX))
        raise ValueError(f"unknown comparison medium {comparison_medium!r}; known: {known}") from exc


def medium_where(comparison_medium: str) -> dict:
    """Chroma equality filter. Talk does not search blog, LinkedIn, or X."""
    return {"medium": index_medium_for(comparison_medium)}


def query_excerpts(collection, embedding: list[float], k: int, where: dict | None = None) -> list[dict]:
    """Return structured hits. An empty match list stays empty."""
    kwargs = {
        "query_embeddings": [embedding],
        "n_results": k,
        "include": ["documents", "metadatas", "distances"],
    }
    if where is not None:
        kwargs["where"] = where
    result = collection.query(**kwargs)
    documents = (result.get("documents") or [[]])[0] or []
    metadatas = (result.get("metadatas") or [[]])[0] or []
    ids = (result.get("ids") or [[]])[0] or []
    distances = (result.get("distances") or [[]])[0] or []
    if len(ids) < len(documents):
        ids = list(ids) + [None] * (len(documents) - len(ids))
    if len(metadatas) < len(documents):
        metadatas = list(metadatas) + [{}] * (len(documents) - len(metadatas))
    if len(distances) < len(documents):
        distances = list(distances) + [None] * (len(documents) - len(distances))
    hits = []
    for doc_id, text, meta, distance in zip(ids, documents, metadatas, distances):
        if not text:
            continue
        meta = meta or {}
        hits.append(
            {
                "id": doc_id,
                "text": text,
                "medium": meta.get("medium"),
                "group_id": meta.get("group_id"),
                "row_id": meta.get("row_id"),
                "distance": distance,
            }
        )
    return hits


def longest_shared_word_span(answer: str, excerpt: str) -> int:
    """Length of the longest contiguous shared word sequence. Not a voice score."""
    left = (answer or "").lower().split()
    right = (excerpt or "").lower().split()
    if not left or not right:
        return 0
    best = 0
    previous = [0] * (len(right) + 1)
    for word in left:
        current = [0]
        for index, other in enumerate(right):
            if word == other:
                span = previous[index] + 1
                current.append(span)
                if span > best:
                    best = span
            else:
                current.append(0)
        previous = current
    return best


def evidence_review_rows(traces: list[dict], adapter: str) -> list[dict]:
    """Human sheet. on_topic and claim_support stay empty for a reviewer."""
    rows = []
    for trace in traces:
        for hit in trace.get("retrieval_hits") or []:
            text = hit.get("text") or ""
            rows.append(
                {
                    "id": trace.get("id"),
                    "medium": trace.get("medium"),
                    "adapter": adapter,
                    "hit_id": hit.get("id"),
                    "shared_word_span": longest_shared_word_span(trace.get("answer") or "", text),
                    "on_topic": "",
                    "claim_support": "",
                }
            )
    return rows


def select_fixed_exemplars(
    dataset_rows: list[dict],
    assignments: list[dict],
    blocked_groups: set[str],
    per_medium: int = 2,
) -> dict:
    """Earliest complete train-owned posts. Review candidates, not a loaded prompt."""
    if per_medium < 1:
        raise ValueError("per_medium must be positive")
    records = assignment_rows(assignments)
    if len(records) != len(dataset_rows):
        raise ValueError(
            f"assignment rows ({len(records)}) != dataset rows ({len(dataset_rows)})"
        )
    chosen = {platform: [] for platform in STYLE_PLATFORMS}
    for row, rec in zip(dataset_rows, records):
        platform = rec.get("source_platform")
        if platform not in chosen or len(chosen[platform]) >= per_medium:
            continue
        if rec.get("split") != "train" or rec.get("sft_role") != "unchanged":
            continue
        if rec.get("group_id") in blocked_groups:
            continue
        text = (row.get("output") or "").strip()
        if not text:
            continue
        chosen[platform].append(
            {
                "row_id": int(rec["row_index"]),
                "group_id": rec["group_id"],
                "medium": platform,
                "comparison_medium": "talk" if platform == "youtube_jason" else platform,
                "sft_role": "unchanged",
                "source_file": rec.get("source_file"),
                "source_line": rec.get("source_line"),
                "base_title": rec.get("base_title"),
                "text": text,
                "use": "voice_only_not_evidence",
            }
        )
    return {
        "status": "review_candidate",
        "loaded_by_default": False,
        "per_medium": per_medium,
        "selection": (
            "Earliest train rows with sft_role unchanged. "
            "Gold-overlap families are excluded. Not loaded unless --fixed-style is set."
        ),
        "gold_overlap_excluded": True,
        "empty_platforms": [platform for platform, items in chosen.items() if not items],
        "exemplars": chosen,
    }


def write_fixed_style_candidates(path: Path | None = None) -> dict:
    """Write the review file. Callers must pass ``--fixed-style`` before it is read."""
    path = path or DEFAULT_STYLE_PATH
    dataset = load_jsonl(DEFAULT_DATASET)
    assignments = load_jsonl(DEFAULT_ASSIGNMENTS)
    dispositions = load_jsonl(DEFAULT_GOLD_DISPOSITIONS)
    payload = select_fixed_exemplars(
        dataset, assignments, blocked_group_ids(assignments, dispositions)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def format_voice_exemplars(exemplars: list[str] | None) -> str:
    """Voice slot. Empty input yields an empty string so the prompt stays unchanged."""
    cleaned = [text.strip() for text in (exemplars or []) if text and text.strip()]
    if not cleaned:
        return ""
    lines = [
        "Style examples from train-owned source text.",
        "Use them for voice only. They are not evidence for claims.",
        "Do not copy long passages verbatim.",
    ]
    for number, text in enumerate(cleaned, 1):
        lines.append(f"[Style {number}]\n{text}")
    return "\n".join(lines)
