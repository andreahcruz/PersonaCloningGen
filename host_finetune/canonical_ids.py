"""Pure identity helpers for a future canonical pipeline.

No file IO. These functions do not read datasets, assign splits, or replace
``split_groups`` group ids.

``source_id`` names the external item (article, tweet, post, video).
``raw_record_id`` names one ingested capture of that item. It is a hash of the
canonical content-bearing fields. It is not a forensic hash of every byte in
the original JSONL record. Scrape timestamps and other declared volatile fields
are excluded. ``raw_payload_sha256`` is the hash of the stored JSON object
bytes when the caller still has those bytes.

A second transcript of the same video keeps ``source_id`` and receives a new
``raw_record_id``.
"""
from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import dataclass

# Recorded on rows. A bump changes a derived id only when that id's inputs
# include the constant.
RAW_SCHEMA_VERSION = "raw-v1"
IDENTITY_VERSION = "identity-v2"
NORMALIZATION_VERSION = "normalize-v1"
CLEANING_VERSION = "clean-scraped-v1"
SFT_BUILD_VERSION = "sft-build-v1"
CONTEXT_FIT_VERSION = "fit512-keepbreaks-v1"
BALANCE_VERSION = "balance-v1"
GROUP_VERSION = "split-groups-v1"
SPLIT_VERSION = "split-groups-v1"
RELABEL_VERSION = "relabel-continuations-v1"
RAG_GOVERNANCE_VERSION = "rag-governance-v1"
EMBEDDING_VERSION = "nomic-embed-text-cpu-cosine-v1"

SOURCE_ID_PREFIX = "source-v2"
RAW_RECORD_PREFIX = "raw-record-v1"
DOCUMENT_PREFIX = "document-v1"
MODEL_ROW_PREFIX = "model-row-v1"
RELABEL_ROW_PREFIX = "relabel-row-v1"

YOUTUBE_MEDIA = frozenset({"youtube_jason", "youtube_saastr"})

# Payload fields for raw_record_id. Volatile scrape metadata is omitted.
# Every listed field is emitted. A missing key and JSON null both become null.
RAW_RECORD_FIELDS = {
    "blog": ("content", "date", "tags", "title", "url"),
    "x": (
        "created_at",
        "id",
        "is_reply",
        "is_repost_or_quote",
        "text",
        "url",
        "username",
    ),
    "linkedin": ("author", "author_source", "content", "format", "url", "urn"),
    "youtube_jason": (
        "channel",
        "description",
        "transcript_error",
        "transcript_language",
        "transcript_segments_count",
        "transcript_source",
        "transcript_text",
        "upload_date",
        "upload_date_iso",
        "video_id",
        "video_title",
        "video_url",
    ),
    "youtube_saastr": (
        "channel",
        "description",
        "transcript_error",
        "transcript_language",
        "transcript_segments_count",
        "transcript_source",
        "transcript_text",
        "upload_date",
        "upload_date_iso",
        "video_id",
        "video_title",
        "video_url",
    ),
}

PLATFORM_URL = "platform_url"
PLATFORM_TWEET_ID = "platform_tweet_id"
PLATFORM_URN = "platform_urn"
PLATFORM_VIDEO_ID = "platform_video_id"
CONTENT_FALLBACK = "content_fallback"


@dataclass(frozen=True)
class SourceIdentity:
    """External identity. ``external_id`` is null for a content fallback."""

    source_id: str
    source_identity_kind: str
    external_id: str | None


def _digest(material: str) -> str:
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _join(*parts: str) -> str:
    return "\n".join(parts)


def _nfc(value: str) -> str:
    return unicodedata.normalize("NFC", value)


def canonical_url(url: str) -> str:
    """Lowercase scheme and host, drop the fragment, and strip one trailing slash."""
    text = _nfc((url or "").strip())
    if not text:
        return ""
    text = text.split("#", 1)[0].strip()
    if "://" not in text:
        return text.rstrip("/")
    scheme, rest = text.split("://", 1)
    host, separator, path = rest.partition("/")
    path = path.rstrip("/")
    if separator and path:
        return f"{scheme.lower()}://{host.lower()}/{path}"
    return f"{scheme.lower()}://{host.lower()}"


def source_id(medium: str, kind: str, key: str) -> str:
    """Hash an external key. Capture differences belong in ``raw_record_id``."""
    if not medium or not kind or not str(key).strip():
        raise ValueError("source identity requires medium, kind, and key")
    return _digest(_join(SOURCE_ID_PREFIX, medium, kind, str(key).strip()))


def _canon_value(value):
    """Normalize one payload value. Key order is handled by the serializer."""
    if value is None:
        return None
    if isinstance(value, str):
        return _nfc(value)
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value != value:
            return None
        return value
    if isinstance(value, list):
        return [_canon_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _canon_value(item) for key, item in value.items()}
    return _nfc(str(value))


def canonical_raw_payload(medium: str, record: dict) -> dict:
    """Schema fields only. Missing and null are both null. Extra keys are dropped."""
    fields = RAW_RECORD_FIELDS.get(medium)
    if fields is None:
        raise ValueError(f"unknown medium {medium}")
    return {field: _canon_value(record.get(field)) for field in fields}


def canonical_raw_json(medium: str, record: dict) -> str:
    """Compact UTF-8 JSON. Sorted keys, no ASCII escaping, no record newline."""
    payload = canonical_raw_payload(medium, record)
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def raw_record_id(medium: str, record: dict) -> str:
    """Canonical capture identity. Not a byte-for-byte hash of the raw JSONL record.

    Volatile scrape fields are omitted. Use ``raw_payload_sha256`` for the stored
    JSON object bytes.
    """
    return _digest(_join(RAW_RECORD_PREFIX, medium, canonical_raw_json(medium, record)))


def raw_payload_sha256(record_bytes: bytes) -> str:
    """SHA-256 of one stored JSON object, excluding its JSONL line terminator.

    The caller must pass the exact UTF-8 bytes taken from the file. This function
    does not rebuild those bytes from a dict. A change to ``scraped_at`` changes
    this hash and does not, by itself, change ``raw_record_id``.
    """
    if not isinstance(record_bytes, (bytes, bytearray)):
        raise TypeError("raw_payload_sha256 requires the stored record bytes")
    return hashlib.sha256(bytes(record_bytes)).hexdigest()


def blog_source_identity(url: str) -> SourceIdentity:
    external = canonical_url(url)
    return SourceIdentity(source_id("blog", PLATFORM_URL, external), PLATFORM_URL, external)


def x_source_identity(tweet_id: str) -> SourceIdentity:
    external = str(tweet_id).strip()
    return SourceIdentity(source_id("x", PLATFORM_TWEET_ID, external), PLATFORM_TWEET_ID, external)


def linkedin_source_identity(*, urn: str = "", url: str = "", raw_content: str = "") -> SourceIdentity:
    """URN, then URL. A body hash is a typed fallback, not a platform id."""
    if urn.strip():
        external = urn.strip()
        return SourceIdentity(source_id("linkedin", PLATFORM_URN, external), PLATFORM_URN, external)
    if url.strip():
        external = canonical_url(url)
        return SourceIdentity(source_id("linkedin", PLATFORM_URL, external), PLATFORM_URL, external)
    if raw_content:
        fingerprint = _digest(_nfc(raw_content))
        return SourceIdentity(
            source_id("linkedin", CONTENT_FALLBACK, fingerprint),
            CONTENT_FALLBACK,
            None,
        )
    raise ValueError("linkedin record has no urn, url, or raw content")


def youtube_source_identity(medium: str, video_id: str) -> SourceIdentity:
    """Channel family plus video id. Transcript capture is not part of source_id."""
    if medium not in YOUTUBE_MEDIA:
        raise ValueError(f"unsupported youtube medium {medium}")
    external = video_id.strip()
    return SourceIdentity(source_id(medium, PLATFORM_VIDEO_ID, external), PLATFORM_VIDEO_ID, external)


def source_identity_for_raw(medium: str, record: dict) -> SourceIdentity:
    """External identity from one raw record. Scrape timestamps are not inputs."""
    if medium == "blog":
        return blog_source_identity(str(record.get("url") or ""))
    if medium == "x":
        tweet_id = record.get("id")
        if tweet_id is None or not str(tweet_id).strip():
            raise ValueError("x record has no tweet id")
        return x_source_identity(str(tweet_id))
    if medium == "linkedin":
        return linkedin_source_identity(
            urn=str(record.get("urn") or ""),
            url=str(record.get("url") or ""),
            raw_content=str(record.get("content") or ""),
        )
    if medium in YOUTUBE_MEDIA:
        return youtube_source_identity(medium, str(record.get("video_id") or ""))
    raise ValueError(f"unknown medium {medium}")


def document_id(raw_record_id_value: str) -> str:
    """L1 document for exactly one capture. v1 does not collapse captures by source_id."""
    if not raw_record_id_value:
        raise ValueError("document_id requires a raw_record_id")
    return _digest(_join(DOCUMENT_PREFIX, raw_record_id_value))


def model_row_id(
    document_id_value: str,
    *,
    fit_version: str,
    chunk_ordinal: int,
    output: str,
) -> str:
    """Fitted-chunk identity. Traces to one capture through ``document_id``.

    Split, group, balance, and RAG eligibility are not inputs. Balance may drop
    a row. It does not rename the rows that remain.
    """
    if chunk_ordinal < 0:
        raise ValueError("chunk_ordinal must be >= 0")
    if not document_id_value or not fit_version:
        raise ValueError("model_row_id requires document_id and fit_version")
    return _digest(
        _join(
            MODEL_ROW_PREFIX,
            document_id_value,
            fit_version,
            str(chunk_ordinal),
            _digest(output),
        )
    )


def relabel_row_id(
    model_row_id_value: str,
    *,
    relabel_version: str,
    sft_role: str,
    instruction: str,
) -> str:
    """Relabel identity. RAG exclusion reasons are not inputs."""
    if not model_row_id_value or not relabel_version or not sft_role:
        raise ValueError("relabel_row_id requires model row, version, and role")
    return _digest(
        _join(
            RELABEL_ROW_PREFIX,
            model_row_id_value,
            relabel_version,
            sft_role,
            _digest(instruction),
        )
    )


def rag_row_id(relabel_row_id_value: str) -> str:
    """Governance does not mint a second content id for an eligible Relabel row."""
    if not relabel_row_id_value:
        raise ValueError("rag_row_id requires a relabel_row_id")
    return relabel_row_id_value


def experiment_row_id(row_index: int) -> str:
    """Frozen index identity. This is not a raw-document id."""
    if row_index < 0:
        raise ValueError("row_index must be >= 0")
    return f"train_{row_index}"
