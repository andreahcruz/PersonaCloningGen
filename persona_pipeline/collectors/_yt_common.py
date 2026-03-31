"""
Shared helpers for YouTube transcript collectors.

Uses yt-dlp (via subprocess) to list channel videos and download
auto-generated captions in WebVTT format, then parses them to plain text.

Features adapted from scrape_transcripts.py:
- --write-info-json for upload_date / description metadata
- Per-video retry with exponential backoff
- JSONL streaming (crash-resilient) + JSON array for Spark
- Resume via seen_video_ids cache
- Error recording instead of silent skipping
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = str(Path(__file__).resolve().parents[2])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from persona_pipeline.config.config import (
    PERSONA_USERNAME,
    YT_COOKIES_PATH,
    YT_DELAY_SECONDS,
    YT_MAX_VIDEOS,
    MIN_WORD_COUNT,
    RAW_DIR,
    persona_raw_path,
)

_PYTHON = sys.executable

sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

MAX_TRANSCRIPT_RETRIES = 2
RETRY_BACKOFF_SECONDS = 3.0


def _cookies_args() -> list[str]:
    """Return ['--cookies', path] if a cookies file is configured, else []."""
    if YT_COOKIES_PATH and os.path.isfile(YT_COOKIES_PATH):
        return ["--cookies", YT_COOKIES_PATH]
    return []


def _video_id_from_entry(entry: dict) -> str | None:
    vid = entry.get("id")
    if vid and re.fullmatch(r"[a-zA-Z0-9_-]{11}", str(vid)):
        return str(vid)
    url = entry.get("webpage_url") or entry.get("url") or ""
    m = re.search(r"[?&]v=([a-zA-Z0-9_-]{11})", url)
    return m.group(1) if m else None


def list_channel_videos(channel_url: str, limit: int = YT_MAX_VIDEOS) -> list[dict]:
    """Use yt-dlp --flat-playlist to enumerate videos on a channel."""
    cmd = [
        _PYTHON, "-m", "yt_dlp",
        "--flat-playlist", "--dump-json",
        "--no-warnings", "--ignore-errors",
        "--playlist-items", f"1-{limit}",
        *_cookies_args(),
        channel_url,
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", timeout=300,
    )
    if result.returncode != 0 and not result.stdout.strip():
        warnings.warn(f"yt-dlp listing failed for {channel_url}: {result.stderr[:500]}")
        return []

    entries: list[dict] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            vid = _video_id_from_entry(obj)
            if vid:
                entries.append({
                    "video_id": vid,
                    "title": obj.get("title") or "",
                    "url": obj.get("webpage_url") or f"https://www.youtube.com/watch?v={vid}",
                })
        except json.JSONDecodeError:
            continue
    return entries


def _vtt_to_plain_text(vtt_path: str) -> str:
    """Parse a WebVTT file into deduplicated plain text."""
    with open(vtt_path, encoding="utf-8", errors="replace") as f:
        raw = f.read()

    if raw.startswith("WEBVTT"):
        parts = raw.split("\n\n", 1)
        raw = parts[1] if len(parts) > 1 else ""

    lines_out: list[str] = []
    prev: str | None = None
    for line in raw.splitlines():
        line = line.strip()
        if not line or "-->" in line:
            continue
        if line.startswith(("NOTE", "Kind:", "Language:")):
            continue
        line = re.sub(r"<\d{2}:\d{2}:\d{2}\.\d{3}>", "", line)
        line = re.sub(r"<[^>]+>", "", line).strip()
        if not line or line == prev:
            continue
        lines_out.append(line)
        prev = line
    return " ".join(lines_out)


def _vtt_cue_count(vtt_path: str) -> int:
    with open(vtt_path, encoding="utf-8", errors="replace") as f:
        return sum(1 for line in f if "-->" in line)


def _pick_vtt_file(tmpdir: str) -> str:
    names = [n for n in os.listdir(tmpdir) if n.endswith(".vtt")]
    if not names:
        raise FileNotFoundError("no .vtt files after yt-dlp")

    def rank(n: str) -> tuple[int, str]:
        nl = n.lower()
        if nl.endswith(".en.vtt") and "en-orig" not in nl:
            return (0, n)
        if "en-orig" in nl:
            return (1, n)
        return (2, n)

    names.sort(key=rank)
    return os.path.join(tmpdir, names[0])


def _upload_date_to_iso(yyyymmdd: str | None) -> str | None:
    if not yyyymmdd or len(yyyymmdd) != 8 or not yyyymmdd.isdigit():
        return None
    return f"{yyyymmdd[:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:8]}"


def _extract_info_json(tmpdir: str, video_id: str) -> dict[str, Any]:
    """Read metadata from the .info.json file yt-dlp writes."""
    info_path = os.path.join(tmpdir, f"{video_id}.info.json")
    meta: dict[str, Any] = {"upload_date": None, "upload_date_iso": None, "description": None}
    if not os.path.isfile(info_path):
        return meta
    try:
        with open(info_path, encoding="utf-8") as f:
            info = json.load(f)
        if not isinstance(info, dict):
            return meta
        ud = info.get("upload_date")
        if isinstance(ud, str) and len(ud) == 8 and ud.isdigit():
            meta["upload_date"] = ud
            meta["upload_date_iso"] = _upload_date_to_iso(ud)
        d = info.get("description")
        if isinstance(d, str):
            meta["description"] = d
    except (json.JSONDecodeError, OSError):
        pass
    return meta


def fetch_transcript(
    video_id: str,
) -> tuple[str | None, int, dict[str, Any]]:
    """
    Download auto-captions + info.json for one video.

    Returns (transcript_text_or_None, cue_count, metadata_dict).
    Raises on yt-dlp failure so the caller can retry.
    """
    tmpdir = tempfile.mkdtemp(prefix="ytdlp_subs_")
    try:
        url = f"https://www.youtube.com/watch?v={video_id}"
        cmd = [
            _PYTHON, "-m", "yt_dlp",
            "--no-warnings", "--skip-download",
            "--write-auto-subs",
            "--write-info-json",
            "--sub-langs", "en.*",
            "--sub-format", "vtt",
            "-o", "%(id)s.%(ext)s",
            *_cookies_args(),
            url,
        ]
        completed = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=300, cwd=tmpdir,
        )

        meta = _extract_info_json(tmpdir, video_id)

        if completed.returncode != 0:
            tail = (completed.stderr or "")[-2000:] + (completed.stdout or "")[-1000:]
            raise RuntimeError(f"yt-dlp exit {completed.returncode}: {tail}")

        vtt_path = _pick_vtt_file(tmpdir)
        text = _vtt_to_plain_text(vtt_path)
        cues = _vtt_cue_count(vtt_path)
        return text, cues, meta
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _fetch_metadata_only(video_id: str) -> dict[str, Any]:
    """Fallback: grab just the info.json when subtitles fail."""
    tmpdir = tempfile.mkdtemp(prefix="ytdlp_meta_")
    try:
        url = f"https://www.youtube.com/watch?v={video_id}"
        cmd = [
            _PYTHON, "-m", "yt_dlp",
            "--no-warnings", "--skip-download",
            "--write-info-json",
            "-o", "%(id)s.%(ext)s",
            *_cookies_args(),
            url,
        ]
        subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=120, cwd=tmpdir,
        )
        return _extract_info_json(tmpdir, video_id)
    except Exception:
        return {"upload_date": None, "upload_date_iso": None, "description": None}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _load_seen(cache_path: str) -> set[str]:
    if not os.path.exists(cache_path):
        return set()
    try:
        with open(cache_path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("video_ids"), list):
            return {str(x) for x in data["video_ids"]}
    except Exception:
        pass
    return set()


def _save_seen(cache_path: str, video_ids: set[str]) -> None:
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    tmp = cache_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"video_ids": sorted(video_ids)}, f, ensure_ascii=False, indent=2)
    os.replace(tmp, cache_path)


def _append_jsonl(path: str, row: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def collect_channel(
    channel_url: str,
    source_label: str,
    output_filename: str,
    limit: int = YT_MAX_VIDEOS,
) -> None:
    """Full pipeline: list videos -> fetch transcripts -> write JSON + JSONL."""
    os.makedirs(str(RAW_DIR), exist_ok=True)

    seen_cache = str(RAW_DIR / f".seen_{output_filename.replace('.json', '')}.json")
    jsonl_path = persona_raw_path(output_filename.replace(".json", ".jsonl"))
    json_path = persona_raw_path(output_filename)

    seen = _load_seen(seen_cache)

    print(f"[{output_filename}] listing videos from {channel_url} (limit={limit})")
    videos = list_channel_videos(channel_url, limit=limit)
    print(f"[{output_filename}] found {len(videos)} videos")

    if seen:
        print(f"[{output_filename}] resuming — {len(seen)} videos already processed")

    if YT_COOKIES_PATH and os.path.isfile(YT_COOKIES_PATH):
        print(f"[{output_filename}] using cookies from {YT_COOKIES_PATH}")
    else:
        print(f"[{output_filename}] WARNING: no cookies file — YouTube may block caption access")

    items: list[dict] = []
    total_ok = 0
    total_skip = 0
    total_err = 0
    consecutive_misses = 0

    for i, v in enumerate(videos, start=1):
        video_id = v["video_id"]
        title = v["title"]

        if video_id in seen:
            continue

        row: dict[str, Any] = {
            "id": hashlib.sha1(video_id.encode()).hexdigest()[:12],
            "source": source_label,
            "url": v["url"],
            "title": title,
            "text": None,
            "word_count": 0,
            "author": PERSONA_USERNAME,
            "video_id": video_id,
            "channel": channel_url,
            "transcript_source": "yt-dlp",
            "transcript_language": None,
            "transcript_segments_count": None,
            "transcript_error": None,
            "upload_date": None,
            "upload_date_iso": None,
            "description": None,
            "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        }

        err: Exception | None = None
        for attempt in range(MAX_TRANSCRIPT_RETRIES + 1):
            try:
                text, cues, meta = fetch_transcript(video_id)
                row["text"] = text
                row["transcript_language"] = "en"
                row["transcript_segments_count"] = cues
                row.update(meta)
                err = None
                break
            except Exception as e:
                err = e
                if attempt < MAX_TRANSCRIPT_RETRIES:
                    time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))

        if err is not None:
            row["transcript_error"] = f"{type(err).__name__}: {err}"
            fallback_meta = _fetch_metadata_only(video_id)
            row.update({k: v for k, v in fallback_meta.items() if v is not None})
            total_err += 1
            consecutive_misses += 1
            print(f"[{output_filename}] {i}/{len(videos)} ERR: {title[:55]} | {type(err).__name__}")
        elif not row["text"] or len((row["text"] or "").split()) < MIN_WORD_COUNT:
            wc = len((row["text"] or "").split())
            if not row["text"]:
                row["transcript_error"] = "no transcript text returned"
                consecutive_misses += 1
                print(f"[{output_filename}] {i}/{len(videos)} SKIP (no captions): {title[:55]}")
            else:
                consecutive_misses = 0
                print(f"[{output_filename}] {i}/{len(videos)} SKIP (too short, {wc} words): {title[:55]}")
            total_skip += 1
        else:
            row["word_count"] = len(row["text"].split())
            items.append(row)
            consecutive_misses = 0
            total_ok += 1
            print(f"[{output_filename}] {i}/{len(videos)} OK ({row['word_count']} words): {title[:55]}")

        _append_jsonl(jsonl_path, row)
        seen.add(video_id)
        _save_seen(seen_cache, seen)

        if consecutive_misses >= 20 and total_ok == 0:
            print(f"[{output_filename}] first {consecutive_misses} videos had no captions — aborting (check cookies)")
            break

        time.sleep(YT_DELAY_SECONDS)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    print(f"\n[{output_filename}] done: {total_ok} OK, {total_skip} skipped, {total_err} errors")
    print(f"[{output_filename}] JSON  (for Spark): {json_path} ({len(items)} items)")
    print(f"[{output_filename}] JSONL (full log):  {jsonl_path}")
