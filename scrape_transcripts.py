import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from tqdm import tqdm


@dataclass(frozen=True)
class ChannelConfig:
    name: str
    url: str


CHANNELS: Sequence[ChannelConfig] = [
    ChannelConfig(name="Jason M. Lemkin", url="https://www.youtube.com/@Jasonlk/videos"),
    ChannelConfig(name="SaaStr", url="https://www.youtube.com/@Saastr/videos"),
]


KEYWORD_RULES: Sequence[Tuple[str, re.Pattern[str]]] = [
    ("ama", re.compile(r"\bama\b", re.IGNORECASE)),
    ("workshop wednesday", re.compile(r"workshop\s+wednesday", re.IGNORECASE)),
    ("keynote", re.compile(r"\bkeynote\b", re.IGNORECASE)),
    ("10 ways", re.compile(r"\b10\s+ways\b", re.IGNORECASE)),
    (
        "5 interesting learnings",
        re.compile(r"\b5\s+(?:more\s+)?interesting\s+learnings\b", re.IGNORECASE),
    ),
]


def safe_filename(name: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", name.strip())
    return name.strip("_") or "dataset"


def match_title(title: str) -> List[str]:
    matched: List[str] = []
    for rule_name, pattern in KEYWORD_RULES:
        if pattern.search(title or ""):
            matched.append(rule_name)
    return matched


def extract_video_id(video_url_or_id: str) -> str:
    if re.fullmatch(r"[a-zA-Z0-9_-]{11}", video_url_or_id):
        return video_url_or_id
    m = re.search(r"[?&]v=([a-zA-Z0-9_-]{11})", video_url_or_id)
    if m:
        return m.group(1)
    raise ValueError(f"Could not extract video id from: {video_url_or_id}")


def _run_yt_dlp_dump_json(url: str, playlist_items: Optional[str]) -> List[Dict[str, Any]]:
    """
    If playlist_items is None, yt-dlp lists the entire playlist (all channel videos).
    Otherwise use e.g. \"1-50\" for the first 50 entries.
    """
    cmd = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--flat-playlist",
        "--dump-json",
        "--no-warnings",
        "--ignore-errors",
    ]
    if playlist_items is not None:
        cmd.extend(["--playlist-items", playlist_items])
    cmd.append(url)
    completed = subprocess.run(cmd, check=False, capture_output=True, text=True, encoding="utf-8")
    if completed.returncode != 0 and not completed.stdout.strip():
        raise RuntimeError(
            "yt-dlp failed while listing channel videos.\n"
            f"returncode={completed.returncode}\n"
            f"stderr={completed.stderr.strip()[:5000]}\n"
            f"stdout_tail={completed.stdout.strip()[:5000]}"
        )
    entries: List[Dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("id"):
            entries.append(obj)
    return entries


def list_channel_videos(url: str, limit: Optional[int]) -> List[Dict[str, Any]]:
    if limit is None or limit <= 0:
        playlist_items: Optional[str] = None
    else:
        playlist_items = f"1-{limit}"
    entries = _run_yt_dlp_dump_json(url=url, playlist_items=playlist_items)
    normalized: List[Dict[str, Any]] = []
    for e in entries:
        vid = e.get("id")
        if not vid:
            continue
        normalized.append(
            {
                "video_id": str(vid),
                "title": e.get("title") or "",
                "webpage_url": e.get("webpage_url") or f"https://www.youtube.com/watch?v={vid}",
            }
        )
    return normalized


def load_seen_video_ids(cache_path: str) -> Set[str]:
    if not os.path.exists(cache_path):
        return set()
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("video_ids"), list):
            return {str(x) for x in data["video_ids"]}
    except Exception:
        pass
    return set()


def save_seen_video_ids(cache_path: str, video_ids: Set[str]) -> None:
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    payload = {"video_ids": sorted(video_ids)}
    tmp_path = cache_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, cache_path)


def append_jsonl(path: str, row: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def vtt_to_plain_text(vtt_path: str) -> str:
    with open(vtt_path, encoding="utf-8", errors="replace") as f:
        raw = f.read()
    if raw.startswith("WEBVTT"):
        parts = raw.split("\n\n", 1)
        raw = parts[1] if len(parts) > 1 else ""
    lines_out: List[str] = []
    prev: Optional[str] = None
    for line in raw.splitlines():
        line = line.strip()
        if not line or "-->" in line:
            continue
        if line.startswith("NOTE") or line.startswith("Kind:") or line.startswith("Language:"):
            continue
        line = re.sub(r"<\d{2}:\d{2}:\d{2}\.\d{3}>", "", line)
        line = re.sub(r"<[^>]+>", "", line)
        line = line.strip()
        if not line:
            continue
        if line == prev:
            continue
        lines_out.append(line)
        prev = line
    return " ".join(lines_out)


def vtt_cue_count(vtt_path: str) -> int:
    with open(vtt_path, encoding="utf-8", errors="replace") as f:
        return sum(1 for line in f if "-->" in line)


def upload_date_yyyymmdd_to_iso(y: Optional[str]) -> Optional[str]:
    if not y or len(y) != 8 or not y.isdigit():
        return None
    return f"{y[:4]}-{y[4:6]}-{y[6:8]}"


def fetch_info_metadata_ytdlp(
    video_id: str,
    *,
    cookies_path: Optional[str],
    proxy: Optional[str],
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Metadata only (no subtitles): reads {id}.info.json from a skip-download run.
    Returns (upload_date_yyyymmdd, upload_date_iso, description).
    """
    url = f"https://www.youtube.com/watch?v={video_id}"
    tmpdir = tempfile.mkdtemp(prefix="ytdlp_meta_")
    try:
        cmd: List[str] = [
            sys.executable,
            "-m",
            "yt_dlp",
            "--no-warnings",
            "--skip-download",
            "--write-info-json",
            "-o",
            "%(id)s.%(ext)s",
            url,
        ]
        if cookies_path:
            cmd.extend(["--cookies", cookies_path])
        if proxy:
            cmd.extend(["--proxy", proxy])
        completed = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
            cwd=tmpdir,
        )
        if completed.returncode != 0:
            return None, None, None
        info_path = os.path.join(tmpdir, f"{video_id}.info.json")
        if not os.path.isfile(info_path):
            return None, None, None
        with open(info_path, encoding="utf-8") as jf:
            info = json.load(jf)
        if not isinstance(info, dict):
            return None, None, None
        upload_y: Optional[str] = None
        upload_iso: Optional[str] = None
        ud = info.get("upload_date")
        if isinstance(ud, str) and len(ud) == 8 and ud.isdigit():
            upload_y = ud
            upload_iso = upload_date_yyyymmdd_to_iso(ud)
        desc: Optional[str] = None
        d = info.get("description")
        if isinstance(d, str):
            desc = d
        return upload_y, upload_iso, desc
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _pick_vtt_file(tmpdir: str) -> str:
    names = [n for n in os.listdir(tmpdir) if n.endswith(".vtt")]
    if not names:
        raise FileNotFoundError("no .vtt files after yt-dlp")
    # Prefer translated/normalized en over en-orig when both exist
    def rank(n: str) -> Tuple[int, str]:
        nl = n.lower()
        if nl.endswith(".en.vtt") and "en-orig" not in nl:
            return (0, n)
        if "en-orig" in nl:
            return (1, n)
        return (2, n)

    names.sort(key=rank)
    return os.path.join(tmpdir, names[0])


def fetch_transcript_ytdlp(
    video_id: str,
    languages: Sequence[str],
    *,
    cookies_path: Optional[str],
    proxy: Optional[str],
) -> Tuple[str, str, int, Optional[str], Optional[str], Optional[str]]:
    """
    Download auto-captions with yt-dlp, parse WebVTT to plain text.
    Uses --write-info-json (not --dump-json) so subtitle files are still written to disk.

    Returns (language_code, transcript_text, cue_count, upload_date_yyyymmdd, upload_date_iso, description).
    """
    url = f"https://www.youtube.com/watch?v={video_id}"
    base = (languages[0] if languages else "en").strip().split(",")[0].strip()
    base = base.split("-")[0] if base else "en"
    # Match e.g. en + en-orig
    sub_langs = f"{base}.*"

    tmpdir = tempfile.mkdtemp(prefix="ytdlp_subs_")
    try:
        # cwd=tmpdir + short -o template: writes *.vtt and *.info.json here.
        # (--dump-json suppresses writing subtitle files in current yt-dlp.)
        out_tmpl = "%(id)s.%(ext)s"
        cmd: List[str] = [
            sys.executable,
            "-m",
            "yt_dlp",
            "--no-warnings",
            "--skip-download",
            "--write-auto-subs",
            "--write-info-json",
            "--sub-langs",
            sub_langs,
            "--sub-format",
            "vtt",
            "-o",
            out_tmpl,
            url,
        ]
        if cookies_path:
            cmd.extend(["--cookies", cookies_path])
        if proxy:
            cmd.extend(["--proxy", proxy])

        completed = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=300,
            cwd=tmpdir,
        )
        upload_y: Optional[str] = None
        upload_iso: Optional[str] = None
        description: Optional[str] = None
        info_path = os.path.join(tmpdir, f"{video_id}.info.json")
        if os.path.isfile(info_path):
            try:
                with open(info_path, encoding="utf-8") as jf:
                    info = json.load(jf)
                if isinstance(info, dict):
                    ud = info.get("upload_date")
                    if isinstance(ud, str) and len(ud) == 8 and ud.isdigit():
                        upload_y, upload_iso = ud, upload_date_yyyymmdd_to_iso(ud)
                    d = info.get("description")
                    if isinstance(d, str):
                        description = d
            except (json.JSONDecodeError, OSError):
                pass

        if completed.returncode != 0:
            tail = (completed.stderr or "")[-4000:] + (completed.stdout or "")[-2000:]
            raise RuntimeError(f"yt-dlp exit {completed.returncode}: {tail}")

        vtt_path = _pick_vtt_file(tmpdir)
        text = vtt_to_plain_text(vtt_path)
        cues = vtt_cue_count(vtt_path)
        return base, text, cues, upload_y, upload_iso, description
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scrape YouTube transcripts via yt-dlp (WebVTT auto-subs) to JSONL."
    )
    parser.add_argument("--channels", type=str, default="Jason M. Lemkin", help="Comma-separated channel names.")
    parser.add_argument(
        "--all-videos",
        action="store_true",
        help="List and transcribe every video on each selected channel (can take a long time).",
    )
    parser.add_argument(
        "--max-videos-per-channel",
        type=int,
        default=10,
        help="Max videos to fetch per channel. Use 0 with --all-videos or pass 0 alone to mean all videos.",
    )
    parser.add_argument("--output", type=str, default="data/dataset.jsonl", help="JSONL output path.")
    parser.add_argument("--seen-cache", type=str, default="data/seen_video_ids.json", help="Resume cache.")
    parser.add_argument("--resume", action="store_true", help="Skip video_ids in seen-cache.")
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Primary subtitle language code (passed to yt-dlp as LANG.*, e.g. en).",
    )
    parser.add_argument("--enable-title-filter", action="store_true", help="Filter by keyword titles.")
    parser.add_argument("--cookies", type=str, default=None, help="Optional Netscape cookies file for yt-dlp.")
    parser.add_argument("--proxy", type=str, default=None, help="Optional proxy URL for yt-dlp (--proxy).")
    parser.add_argument("--max-transcript-retries", type=int, default=2, help="Retries per video if yt-dlp fails.")
    parser.add_argument("--retry-backoff-seconds", type=float, default=3.0, help="Backoff between retries.")
    parser.add_argument("--sleep-seconds", type=float, default=2.0, help="Sleep after each video.")
    args = parser.parse_args()

    if args.all_videos:
        list_limit: Optional[int] = None
    elif args.max_videos_per_channel <= 0:
        list_limit = None
    else:
        list_limit = args.max_videos_per_channel

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    preferred = [x.strip() for x in args.language.split(",") if x.strip()] or ["en"]

    selected = [x.strip() for x in args.channels.split(",") if x.strip()]
    channels = [ch for ch in CHANNELS if ch.name in set(selected)]
    if not channels:
        raise SystemExit(f"No matching channels for --channels={args.channels}")

    seen: Set[str] = set()
    if args.resume:
        seen = load_seen_video_ids(args.seen_cache)
        print(f"Resuming: {len(seen)} video_ids already processed.")
        save_seen_video_ids(args.seen_cache, seen)

    total_matched = 0
    total_ok = 0
    print(f"Start time (UTC): {datetime.now(timezone.utc).isoformat()}")

    for ch in channels:
        print(f"\nListing videos for channel: {ch.name}")
        videos = list_channel_videos(ch.url, limit=list_limit)
        print(f"Found {len(videos)} videos (pre-filter).")

        for v in tqdm(videos, desc=safe_filename(ch.name), unit="video"):
            try:
                video_id = extract_video_id(v["video_id"])
            except Exception:
                continue
            if args.resume and video_id in seen:
                continue

            title = v.get("title") or ""
            matched_keywords: List[str] = []
            if args.enable_title_filter:
                matched_keywords = match_title(title)
                if not matched_keywords:
                    continue

            total_matched += 1
            page_url = v.get("webpage_url") or f"https://www.youtube.com/watch?v={video_id}"
            row: Dict[str, Any] = {
                "channel": ch.name,
                "video_id": video_id,
                "video_title": title,
                "video_url": page_url,
                "matched_keywords": matched_keywords,
                "transcript_source": "yt-dlp",
                "upload_date": None,
                "upload_date_iso": None,
                "description": None,
                "transcript_text": None,
                "transcript_language": None,
                "transcript_segments_count": None,
                "transcript_error": None,
                "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
                "listing_source": ch.url,
            }

            err: Optional[Exception] = None
            for attempt in range(args.max_transcript_retries + 1):
                try:
                    lang, text, cues, upload_y, upload_iso, desc = fetch_transcript_ytdlp(
                        video_id,
                        preferred,
                        cookies_path=args.cookies,
                        proxy=args.proxy,
                    )
                    row["transcript_text"] = text
                    row["transcript_language"] = lang
                    row["transcript_segments_count"] = cues
                    row["upload_date"] = upload_y
                    row["upload_date_iso"] = upload_iso
                    row["description"] = desc
                    total_ok += 1
                    err = None
                    break
                except Exception as e:
                    err = e
                    if attempt < args.max_transcript_retries:
                        time.sleep(args.retry_backoff_seconds * (attempt + 1))
            if err is not None:
                row["transcript_error"] = f"{type(err).__name__}: {err}"
                uy, uiso, desc = fetch_info_metadata_ytdlp(
                    video_id,
                    cookies_path=args.cookies,
                    proxy=args.proxy,
                )
                if uy:
                    row["upload_date"] = uy
                    row["upload_date_iso"] = uiso
                if desc is not None:
                    row["description"] = desc

            append_jsonl(args.output, row)
            seen.add(video_id)
            if args.resume:
                save_seen_video_ids(args.seen_cache, seen)
            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)

    print(f"\nDone. matched_videos={total_matched}, with_transcript_text={total_ok}")
    print(f"Dataset: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
