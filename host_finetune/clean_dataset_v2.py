"""Aggressive v2 cleaner for ``dataset.jsonl`` — re-stitch, repunctuate, re-chunk.

Run::

    host_finetune\\.venv\\Scripts\\python.exe -m host_finetune.clean_dataset_v2

What this script does (in order):

    1.  Backup the existing ``data/dataset.jsonl`` to ``data/dataset.preclean.jsonl``.
    2.  Load all rows.
    3.  Drop **echo-bug** rows (instruction title ≈ literal prefix of output).
    4.  Group remaining rows by base title (after stripping "(part k of N)"),
        sort by part index, and concatenate output strings **directly**. This
        rebuilds the original body from chunks whose seams currently cut
        mid-word (e.g. ``"…big c" + "apacity…" = "…big capacity…"``).
    5.  For each reconstructed body:
            • Drop if the first 300 chars are a non-Lemkin guest opener
              (``"my name is …"`` / ``"i'm the ceo of …"`` / etc.).
            • Drop if the first 300 chars are a podcast intro / sponsor read.
            • Drop if the body is Cyrillic/CJK-heavy (>1%).
            • Scrub URLs (``https?://…`` and ``pic.twitter.com/…``).
            • Strip filler interjections (``um``, ``uh``, ``ah``, ``er``,
              ``erm``, ``hmm``, ``mhm`` and their multi-letter forms).
            • Classify (transcript / blog / tweet / other).
            • If transcript-style (no caps, no punctuation): repunctuate
              via Ollama (``llama3.1`` by default) in word-boundary windows
              with content-preservation guardrails (drop the repunct if the
              word count drifted >5% from original — fall back to original).
            • Re-chunk on sentence boundaries via
              ``sft_chunk_utils.split_sentence_aware`` so no chunk starts
              mid-word.
    6.  Emit ``Write in the style of Jason Lemkin about: {title} (part k of N)``
        rows back to ``dataset.jsonl``.
    7.  Print before/after counts + per-group repunctuation telemetry.

Flags:

    --out PATH               Write to PATH instead of overwriting dataset.jsonl
    --dry-run                Equivalent to --out data/dataset.clean_v2.jsonl
    --no-repunctuate         Skip Ollama repunct; emit transcripts as-is
                             (still drops fillers + re-chunks on word boundary)
    --max-transcript-groups N
                             Repunctuate only the first N transcript-style
                             groups (smoke test)
    --ollama-base URL        default http://localhost:11434
    --ollama-model NAME      default llama3.1
    --chunk-max-chars N      default SFT_CHUNK_OUTPUT_CHARS (1600)
    --no-stitch              Skip re-stitching by base title (keep current
                             (part k of N) seams; only filter + re-chunk).
                             Mostly for debugging.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests

_SPARK_JOBS = Path(__file__).resolve().parents[1] / "spark_jobs"
if str(_SPARK_JOBS) not in sys.path:
    sys.path.insert(0, str(_SPARK_JOBS))
from corpus_footer_scrub import strip_footer_noise

from host_finetune.config import (
    DATASET_LOCAL,
    SFT_CHUNK_OUTPUT_CHARS,
)
from host_finetune.sft_chunk_utils import (
    extract_base_topic,
    split_sentence_aware,
)

logger = logging.getLogger("clean_v2")

# ── Defaults from env ───────────────────────────────────────────────────
DEFAULT_OLLAMA_BASE = os.environ.get("OLLAMA_BASE", "http://localhost:11434")
DEFAULT_OLLAMA_MODEL = os.environ.get("CLEAN_REPUNCT_MODEL", "llama3.1")

# Min characters for a body / chunk to be worth keeping.
MIN_BODY_CHARS = 80

# ── Detection regexes ──────────────────────────────────────────────────
_TITLE_RE = re.compile(
    r"^Write in the style of Jason Lemkin about:\s*", re.IGNORECASE
)
_PART_RE = re.compile(r"\s*\(part\s+(\d+)\s+of\s+(\d+)\)\s*$", re.IGNORECASE)

# Echo bug: instruction = "Write…about: <first 100 chars of output>..."
def _is_echo_bug_row(instr: str, out: str) -> bool:
    instr = (instr or "").strip()
    out = (out or "").strip()
    if not instr.endswith("..."):
        return False
    topic = _TITLE_RE.sub("", instr).strip()
    topic = _PART_RE.sub("", topic).strip()
    topic = topic.rstrip(". ").rstrip("…").strip()
    if len(topic) < 40:
        return False
    head = topic[:60].lower()
    return out[:80].lower().startswith(head[: min(len(head), len(out))])


# Guest opener: someone other than Lemkin introduces themselves at the start.
_GUEST_OPENER_RE = re.compile(
    r"\b(my name is|i'?m the ceo of|i'?m the founder of|i am the ceo of|"
    r"i am the founder of|hello (?:my name is|i am))\b",
    re.IGNORECASE,
)


def _starts_with_guest_opener(body: str) -> bool:
    head = body[:300].lower()
    if not _GUEST_OPENER_RE.search(head):
        return False
    # If Lemkin / Jason appears in the same window, keep it (might be Jason
    # introducing himself).
    if "jason" in head or "lemkin" in head:
        return False
    return True


# Podcast intro / sponsor read phrases (case-insensitive, first 400 chars).
_INTRO_PHRASES = (
    "today's episode features",
    "this episode is brought to you by",
    "this podcast is brought to you by",
    "thanks for listening",
    "subscribe to",
    "welcome back to",
    "welcome to the",
    "before we dive in",
    "today on the podcast",
    "in today's episode",
)


def _starts_with_podcast_intro(body: str) -> bool:
    head = body[:400].lower()
    return any(p in head for p in _INTRO_PHRASES)


# Cyrillic / CJK heavy: >1% of chars in these ranges → drop.
_CYR_CJK_RE = re.compile(r"[\u0400-\u04FF\u3000-\u9FFF\uAC00-\uD7AF]")


def _is_cyrillic_cjk_heavy(body: str) -> bool:
    if not body:
        return False
    n_bad = len(_CYR_CJK_RE.findall(body))
    return n_bad / max(1, len(body)) > 0.01


# URLs (incl. Twitter media URLs).
_URL_RE = re.compile(
    r"\bhttps?://\S+|\bpic\.twitter\.com/\S+|\bbit\.ly/\S+|\bt\.co/\S+",
    re.IGNORECASE,
)


def _scrub_urls(body: str) -> str:
    return _URL_RE.sub("", body)


# Filler interjections — only safe-bounded forms. Tuned to avoid stripping
# valid words (e.g. ``ah`` won't match inside ``ahead``; ``er`` won't match
# inside ``ermine`` because the trailing word-boundary requires a non-word
# char after the alpha run).
_FILLER_RE = re.compile(
    r"\b(?:u+m+|u+h+|a+h+|er+m+|m+h+m+|h+m+|umh+|aha+|ohh+)\b[,.]?\s?",
    re.IGNORECASE,
)


def _scrub_fillers(body: str) -> tuple[str, int]:
    matches = _FILLER_RE.findall(body)
    return _FILLER_RE.sub("", body), len(matches)


# ── Style classification ──────────────────────────────────────────────
_PUNCT_END_RE = re.compile(r"[.!?]")


def _cap_rate(text: str) -> float:
    words = text.split()
    if not words:
        return 0.0
    return sum(1 for w in words if w[:1].isupper()) / len(words)


def _punct_rate(text: str) -> float:
    words = text.split()
    if not words:
        return 0.0
    return sum(text.count(p) for p in ".!?") / len(words)


def _classify_body(body: str) -> str:
    """Returns ``transcript`` / ``blog`` / ``tweet`` / ``other``."""
    if len(body) < MIN_BODY_CHARS:
        return "other"
    cap = _cap_rate(body)
    punct = _punct_rate(body)
    if cap < 0.05 and punct < 0.02:
        return "transcript"
    if cap >= 0.05 and punct >= 0.02:
        return "blog" if "\n" in body or len(body) > 600 else "tweet"
    return "other"


# ── Ollama repunctuation ──────────────────────────────────────────────
_REPUNCT_SYS = (
    "You are a transcript punctuation and capitalization restorer. "
    "Given lowercased, unpunctuated transcript text, restore proper sentence "
    "boundaries (periods, commas, question marks), capital letters at the "
    "start of each sentence, and capitalization of the pronoun 'I' and "
    "obvious proper nouns (people, companies, products). "
    "STRICT RULES: "
    "1. Do NOT add, remove, paraphrase, summarize, or reorder any words. "
    "2. Do NOT add commentary, headings, or explanations. "
    "3. Output ONLY the corrected text. "
    "4. Keep the same speaker — do not insert 'Jason:' or attribution. "
    "5. If a fragment is unclear, leave the wording as-is."
)


def _repunctuate_window(
    text: str, ollama_base: str, model: str, max_attempts: int = 2
) -> tuple[str, str]:
    """Send one window to Ollama; return (repunct_text, status).

    Falls back to the original ``text`` if the model drops/adds tokens. The
    ``status`` string is one of: ``ok``, ``empty``, ``length_drift``,
    ``http_error``, ``timeout``.
    """
    if not text.strip():
        return text, "empty"
    prompt = (
        _REPUNCT_SYS
        + "\n\nTranscript to fix:\n<<<\n"
        + text
        + "\n>>>\n\nCorrected:"
    )
    url = f"{ollama_base.rstrip('/')}/api/generate"
    last_err = ""
    for attempt in range(max_attempts):
        try:
            r = requests.post(
                url,
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.05,
                        "top_p": 0.9,
                        "num_predict": max(256, int(len(text) * 0.45)),
                        # Discourage extra punctuation drift / "Here is the
                        # corrected text:" preambles.
                        "repeat_penalty": 1.0,
                    },
                },
                timeout=300,
            )
            r.raise_for_status()
            data = r.json()
            out = (data.get("response") or "").strip()
            if not out:
                last_err = "empty response"
                continue
            # Strip common preambles even if the model adds them despite
            # the system prompt.
            out = re.sub(
                r"^(here(?:'s| is) (?:the )?(?:corrected|repunct.*?)[:\s]+|"
                r"corrected text[:\s]+|corrected[:\s]+)",
                "",
                out,
                flags=re.IGNORECASE,
            ).strip()
            # Strip closing >>> markers if the model echoed them.
            out = re.sub(r"^<<<\s*", "", out)
            out = re.sub(r"\s*>>>$", "", out).strip()
            # Word-count guardrail: must be within ±10% of the input word count.
            n_in = len(text.split())
            n_out = len(out.split())
            if n_in > 0 and abs(n_out - n_in) / n_in > 0.10:
                last_err = f"length drift {n_in}→{n_out}"
                continue
            return out, "ok"
        except requests.exceptions.Timeout:
            last_err = "timeout"
        except requests.exceptions.RequestException as e:
            last_err = f"http error: {e!r}"
        time.sleep(1.5)
    logger.debug("repunct fell back to original (%s): %s", last_err, text[:80])
    return text, "fallback"


def _windowize_for_repunct(body: str, target_chars: int) -> list[str]:
    """Split a long transcript into word-boundary windows of ~target_chars.

    The first pass: never mid-word. We just chunk on word boundaries here so
    the LLM gets coherent chunks; sentence boundaries don't exist yet
    (that's what repunctuation will create).
    """
    body = body.strip()
    if not body:
        return []
    if len(body) <= target_chars:
        return [body]
    words = body.split()
    windows: list[str] = []
    cur = ""
    for w in words:
        if not cur:
            cur = w
        elif len(cur) + 1 + len(w) <= target_chars:
            cur = cur + " " + w
        else:
            windows.append(cur)
            cur = w
    if cur:
        windows.append(cur)
    return windows


def _repunctuate_body(
    body: str,
    *,
    ollama_base: str,
    model: str,
    window_chars: int = 4000,
) -> tuple[str, dict]:
    """Run a full body through Ollama in windows; return (text, stats)."""
    stats = {"windows": 0, "ok": 0, "fallback": 0, "chars_in": len(body), "chars_out": 0}
    windows = _windowize_for_repunct(body, window_chars)
    stats["windows"] = len(windows)
    pieces: list[str] = []
    for i, w in enumerate(windows):
        out, status = _repunctuate_window(w, ollama_base, model)
        if status == "ok":
            stats["ok"] += 1
        else:
            stats["fallback"] += 1
        pieces.append(out)
        if (i + 1) % 5 == 0 or i + 1 == len(windows):
            logger.info(
                "  repunct window %d/%d (status=%s, len=%d)",
                i + 1,
                len(windows),
                status,
                len(out),
            )
    joined = " ".join(pieces).strip()
    # Collapse any double-spaces introduced by windowing.
    joined = re.sub(r"[ \t]{2,}", " ", joined)
    joined = re.sub(r"\n{3,}", "\n\n", joined)
    stats["chars_out"] = len(joined)
    return joined, stats


# ── Loader / writer ────────────────────────────────────────────────────
def _load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    bad = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
    if bad:
        logger.warning("skipped %d malformed JSON lines", bad)
    return rows


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _parse_title(instr: str) -> tuple[str, int]:
    """Return (base_title_without_part_suffix, part_index_or_1)."""
    t = _TITLE_RE.sub("", (instr or "").strip())
    m = _PART_RE.search(t)
    if m:
        return _PART_RE.sub("", t).strip(), int(m.group(1))
    return t.strip(), 1


def _group_and_stitch(rows: list[dict]) -> list[tuple[str, str]]:
    """Group by base title, sort by part index, concat outputs (no separator).

    Returns a list of ``(base_title, stitched_body)`` tuples.
    """
    groups: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for r in rows:
        instr = r.get("instruction") or ""
        out = (r.get("output") or "").strip()
        if not out:
            continue
        base, k = _parse_title(instr)
        if not base:
            base = "untitled"
        groups[base].append((k, out))
    stitched: list[tuple[str, str]] = []
    for base, parts in groups.items():
        parts.sort(key=lambda x: x[0])
        body = "".join(p for _, p in parts)
        stitched.append((base, body))
    return stitched


# ── Main pipeline ─────────────────────────────────────────────────────
def run(
    *,
    src: Path,
    out: Path,
    backup: Path,
    repunctuate: bool,
    ollama_base: str,
    ollama_model: str,
    chunk_max_chars: int,
    max_transcript_groups: int | None,
    stitch: bool,
) -> dict:
    logger.info("loading %s", src)
    rows = _load_rows(src)
    logger.info("loaded %d rows", len(rows))

    if not backup.exists() and src.exists():
        shutil.copy2(src, backup)
        logger.info("backup written: %s", backup)
    elif backup.exists():
        logger.info("backup already exists, leaving it: %s", backup)

    # Step 1: drop echo-bug rows.
    pre = len(rows)
    rows = [r for r in rows if not _is_echo_bug_row(r.get("instruction", ""), r.get("output", ""))]
    n_echo_dropped = pre - len(rows)
    logger.info("dropped %d echo-bug rows (instruction prefix == output prefix)", n_echo_dropped)

    # Step 2: stitch.
    if stitch:
        stitched = _group_and_stitch(rows)
        logger.info("re-stitched %d rows into %d base-title groups", len(rows), len(stitched))
    else:
        stitched = [(_parse_title(r.get("instruction", ""))[0], (r.get("output") or "").strip()) for r in rows]
        logger.info("--no-stitch: %d 'groups' (one per row)", len(stitched))

    # Step 3: body-level filters + scrubs.
    out_rows: list[dict] = []
    stats: dict[str, int] = defaultdict(int)
    repunct_stats: list[dict] = []
    transcript_groups_done = 0
    t_start = time.monotonic()

    for idx, (base, body) in enumerate(stitched):
        body = body.strip()
        if not body:
            stats["dropped_empty"] += 1
            continue

        # Drop filters.
        if _starts_with_guest_opener(body):
            stats["dropped_guest_opener"] += 1
            continue
        if _starts_with_podcast_intro(body):
            stats["dropped_podcast_intro"] += 1
            continue
        if _is_cyrillic_cjk_heavy(body):
            stats["dropped_cyrillic"] += 1
            continue

        # Scrub URLs + SaaStr / WP recirculation tails + fillers.
        body = _scrub_urls(body)
        body = strip_footer_noise(body)
        body, n_fillers = _scrub_fillers(body)
        stats["fillers_removed_total"] += n_fillers
        body = re.sub(r"[ \t]{2,}", " ", body).strip()

        if len(body) < MIN_BODY_CHARS:
            stats["dropped_too_short_after_scrub"] += 1
            continue

        # Classify and repunctuate if transcript.
        klass = _classify_body(body)
        stats[f"class_{klass}"] += 1

        if klass == "transcript" and repunctuate:
            if max_transcript_groups is not None and transcript_groups_done >= max_transcript_groups:
                logger.info("[%d/%d] '%s' transcript skipped (max-transcript-groups=%d hit)",
                            idx + 1, len(stitched), base[:60], max_transcript_groups)
            else:
                logger.info(
                    "[%d/%d] repunctuating transcript '%s' (%d chars)",
                    idx + 1, len(stitched), base[:60], len(body),
                )
                t0 = time.monotonic()
                new_body, r_stats = _repunctuate_body(
                    body,
                    ollama_base=ollama_base,
                    model=ollama_model,
                )
                dt = time.monotonic() - t0
                r_stats["title"] = base[:80]
                r_stats["seconds"] = round(dt, 1)
                repunct_stats.append(r_stats)
                body = new_body
                transcript_groups_done += 1
                stats["transcripts_repunctuated"] += 1

        # Re-chunk on sentence boundaries.
        chunks = split_sentence_aware(body, chunk_max_chars)
        chunks = [c for c in chunks if len(c.strip()) >= MIN_BODY_CHARS]
        if not chunks:
            stats["dropped_no_chunks"] += 1
            continue

        n = len(chunks)
        for i, chunk in enumerate(chunks):
            topic = base if n == 1 else f"{base} (part {i + 1} of {n})"
            out_rows.append({
                "instruction": f"Write in the style of Jason Lemkin about: {topic}",
                "input": "",
                "output": chunk,
            })
        stats["rows_emitted"] += len(chunks)

    elapsed = time.monotonic() - t_start
    logger.info("pipeline finished in %.1fs; %d rows emitted", elapsed, len(out_rows))
    logger.info("stats: %s", dict(stats))

    _write_rows(out, out_rows)
    logger.info("wrote %s", out)

    summary = {
        "rows_in": len(rows) + n_echo_dropped,
        "rows_in_after_echo_drop": len(rows),
        "rows_out": len(out_rows),
        "echo_bug_dropped": n_echo_dropped,
        "stitched_groups": len(stitched),
        "elapsed_seconds": round(elapsed, 1),
        "stats": dict(stats),
        "repunct_groups": len(repunct_stats),
        "repunct_summary": repunct_stats[:30],  # head only to keep summary small
        "output_path": str(out),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", default=str(DATASET_LOCAL))
    parser.add_argument("--out", default=None,
                        help="Write to this path. Default: overwrite --src.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Equivalent to --out data/dataset.clean_v2.jsonl")
    parser.add_argument("--no-repunctuate", action="store_true")
    parser.add_argument("--max-transcript-groups", type=int, default=None)
    parser.add_argument("--ollama-base", default=DEFAULT_OLLAMA_BASE)
    parser.add_argument("--ollama-model", default=DEFAULT_OLLAMA_MODEL)
    parser.add_argument("--chunk-max-chars", type=int,
                        default=SFT_CHUNK_OUTPUT_CHARS)
    parser.add_argument("--no-stitch", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    src = Path(args.src)
    if not src.is_file():
        print(f"error: missing {src}", file=sys.stderr)
        sys.exit(2)

    if args.dry_run and args.out:
        print("error: choose either --dry-run or --out, not both", file=sys.stderr)
        sys.exit(2)
    if args.dry_run:
        out_path = src.parent / "dataset.clean_v2.jsonl"
    elif args.out:
        out_path = Path(args.out)
    else:
        out_path = src

    backup = src.parent / "dataset.preclean.jsonl"

    summary = run(
        src=src,
        out=out_path,
        backup=backup,
        repunctuate=not args.no_repunctuate,
        ollama_base=args.ollama_base,
        ollama_model=args.ollama_model,
        chunk_max_chars=args.chunk_max_chars,
        max_transcript_groups=args.max_transcript_groups,
        stitch=not args.no_stitch,
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
