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

# Title-only heuristics for the SaaStr *channel* when you want Jason-heavy talks (frameworks,
# AMAs, keynotes). Not perfect: some interviews match "with Jason Lemkin"; some solo uploads
# have generic titles and will be skipped. Flat playlist listing has no description/speaker tags.
SAASSTR_JASON_PRIMARY_RULES: Sequence[Tuple[str, re.Pattern[str]]] = [
    ("jason lemkin", re.compile(r"\bjason\s+lemkin\b", re.IGNORECASE)),
    ("with jason lemkin", re.compile(r"\bwith\s+jason\s+lemkin\b", re.IGNORECASE)),
    ("jason lemkin on", re.compile(r"\bjason\s+lemkin\s+on\b", re.IGNORECASE)),
    ("10 ways", re.compile(r"\b10\s+ways\b", re.IGNORECASE)),
    ("10 mistakes", re.compile(r"\b10\s+mistakes\b", re.IGNORECASE)),
    ("20 learnings", re.compile(r"\b20\s+learnings\b", re.IGNORECASE)),
    ("20 interesting", re.compile(r"\b20\s+interesting\b", re.IGNORECASE)),
    (
        "5 interesting learnings",
        re.compile(r"\b5\s+(?:more\s+)?interesting\s+learnings\b", re.IGNORECASE),
    ),
    ("workshop wednesday", re.compile(r"workshop\s+wednesday", re.IGNORECASE)),
    ("ama", re.compile(r"\bama\b", re.IGNORECASE)),
    ("ask me anything", re.compile(r"ask\s+me\s+anything", re.IGNORECASE)),
    ("keynote", re.compile(r"\bkeynote\b", re.IGNORECASE)),
    ("state of the cloud", re.compile(r"state\s+of\s+the\s+cloud", re.IGNORECASE)),
    ("state of saas", re.compile(r"state\s+of\s+saas", re.IGNORECASE)),
    ("whats next in saas", re.compile(r"what(?:'|’)?s\s+next\s+in\s+saas", re.IGNORECASE)),
    ("mistakes founders", re.compile(r"mistakes.*\bfounder|\bfounder.*mistakes", re.IGNORECASE)),
    (
        "learnings from metrics",
        re.compile(r"learnings\s+from.*(?:\$|\barr\b|billion|million|revenue)", re.IGNORECASE),
    ),
    ("saasstr annual", re.compile(r"saasstr\s+annual", re.IGNORECASE)),
    ("from zero to", re.compile(r"from\s+\$0\s+to|\$0\s+to\s+\$", re.IGNORECASE)),
    ("office hours", re.compile(r"office\s+hours", re.IGNORECASE)),
    ("burn multiple", re.compile(r"burn\s+multiple", re.IGNORECASE)),
    ("harry stebbings and jason", re.compile(r"harry\s+stebbings.*jason|jason.*harry\s+stebbings", re.IGNORECASE)),
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


def match_saasstr_jason_primary(title: str) -> List[str]:
    matched: List[str] = []
    for rule_name, pattern in SAASSTR_JASON_PRIMARY_RULES:
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


def resolve_uploads_playlist_url(channel_page_url: str) -> str:
    """
    YouTube /@handle/videos (and similar tab URLs) often yield a truncated list in yt-dlp
    (~150 items) because the tab extractor does not always paginate like the uploads playlist.

    The full channel uploads playlist is list=UU + the same suffix as the channel id (UC... -> UU...).
    """
    if re.search(r"[?&]list=UU", channel_page_url, re.IGNORECASE):
        return channel_page_url

    cmd = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--flat-playlist",
        "-J",
        "--playlist-items",
        "1",
        "--no-warnings",
        channel_page_url,
    ]
    completed = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    if completed.returncode != 0 or not (completed.stdout or "").strip():
        return channel_page_url
    try:
        meta = json.loads(completed.stdout.strip())
    except json.JSONDecodeError:
        return channel_page_url
    cid = meta.get("id")
    if isinstance(cid, str) and cid.startswith("UC") and len(cid) == 24:
        uu_list = "UU" + cid[2:]
        return f"https://www.youtube.com/playlist?list={uu_list}"
    return channel_page_url


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


@dataclass
class VttCue:
    start: str
    end: str
    text: str
    speaker: Optional[str] = None


_VTT_TIMESTAMP_LINE = re.compile(
    r"^(\d+:\d{2}:\d{2}\.\d{3})\s+-->\s+(\d+:\d{2}:\d{2}\.\d{3})"
)
_VTT_VOICE_OPEN = re.compile(r"<v\s+([^>]+)>", re.IGNORECASE)


def _strip_vtt_inline_tags(s: str) -> str:
    s = re.sub(r"<\d{2}:\d{2}:\d{2}\.\d{3}>", "", s)
    s = re.sub(r"</?v[^>]*>", "", s, flags=re.IGNORECASE)
    s = re.sub(r"<[^>]+>", "", s)
    return re.sub(r"\s+", " ", s).strip()


def parse_vtt_cues(vtt_path: str) -> List[VttCue]:
    with open(vtt_path, encoding="utf-8", errors="replace") as f:
        raw = f.read()
    if raw.startswith("WEBVTT"):
        parts = raw.split("\n\n", 1)
        raw = parts[1] if len(parts) > 1 else ""
    cues: List[VttCue] = []
    for block in re.split(r"\n\s*\n", raw.strip()):
        lines = [ln.rstrip() for ln in block.splitlines()]
        lines = [ln for ln in lines if ln.strip()]
        if not lines:
            continue
        if lines[0].strip().startswith("NOTE") or lines[0].strip().startswith("STYLE"):
            continue
        ts_line_idx = 0
        if "-->" not in lines[0]:
            if len(lines) > 1 and "-->" in lines[1]:
                ts_line_idx = 1
            else:
                continue
        m = _VTT_TIMESTAMP_LINE.match(lines[ts_line_idx].strip())
        if not m:
            continue
        start, end = m.group(1), m.group(2)
        text_lines = lines[ts_line_idx + 1 :]
        raw_text = " ".join(t.strip() for t in text_lines if t.strip())
        if not raw_text:
            continue
        speaker: Optional[str] = None
        vm = _VTT_VOICE_OPEN.search(raw_text)
        if vm:
            speaker = vm.group(1).strip()
        text_plain = _strip_vtt_inline_tags(raw_text)
        if not text_plain:
            continue
        cues.append(VttCue(start=start, end=end, text=text_plain, speaker=speaker))
    return cues


def vtt_to_plain_text_from_cues(cues: Sequence[VttCue]) -> str:
    lines_out: List[str] = []
    prev: Optional[str] = None
    for c in cues:
        if c.text == prev:
            continue
        lines_out.append(c.text)
        prev = c.text
    return " ".join(lines_out)


def vtt_to_plain_text(vtt_path: str) -> str:
    return vtt_to_plain_text_from_cues(parse_vtt_cues(vtt_path))


def vtt_cue_count(vtt_path: str) -> int:
    return len(parse_vtt_cues(vtt_path))


def _speaker_label_is_jason_lemkin(label: str) -> bool:
    s = label.strip().lower()
    if "lemkin" in s:
        return True
    if re.search(r"\bjason\b", s) and len(s) <= 24:
        return True
    return False


def build_jason_speaker_attribution(
    cues: Sequence[VttCue],
    *,
    channel: str,
    title: str,
    matched_keywords: Sequence[str],
    transcript_ok: bool,
) -> Dict[str, Any]:
    """
    Best-effort Jason Lemkin attribution. YouTube auto-captions usually have no <v> speaker tags;
    then we only have title/channel heuristics (weak).
    """
    labeled = [c for c in cues if c.speaker]
    jason_labeled = [c for c in labeled if c.speaker and _speaker_label_is_jason_lemkin(c.speaker)]

    def heuristic() -> Tuple[str, str]:
        t = (title or "").lower()
        if channel == "Jason M. Lemkin":
            return (
                "assumed_jason_channel_owner",
                "Personal channel; auto-captions rarely include per-speaker tags.",
            )
        if "jason" in t and "lemkin" in t:
            return "likely_jason_featured", "Title names Jason Lemkin."
        if matched_keywords:
            return (
                "possibly_jason_from_title_rules",
                "Title matched Jason-primary / keyword rules; speaker mix not verified from captions.",
            )
        return (
            "unknown",
            "No speaker tags in VTT and no strong title signal.",
        )

    if not transcript_ok or not cues:
        inf, note = heuristic()
        return {
            "method": "heuristic_only" if transcript_ok else "no_transcript",
            "voice_tags_in_captions": False,
            "total_cues": len(cues),
            "cues_with_speaker_label": 0,
            "cues_labeled_jason_lemkin": 0,
            "jason_labeled_fraction_of_labeled": None,
            "jason_labeled_fraction_of_all_cues": None,
            "unique_speaker_labels": [],
            "inferred_primary_speaker": inf,
            "confidence_note": note,
        }

    if labeled:
        frac_l = len(jason_labeled) / len(labeled) if labeled else 0.0
        frac_a = len(jason_labeled) / len(cues) if cues else 0.0
        labels = sorted({c.speaker.strip() for c in labeled if c.speaker})
        if len(jason_labeled) >= 0.65 * len(labeled) and jason_labeled:
            inferred = "jason_dominant_in_labeled_cues"
            note = "Most VTT <v> labels name Jason Lemkin."
        elif jason_labeled:
            inferred = "jason_present_mixed_or_panel"
            note = "Some cues labeled as Jason; other speakers also labeled."
        else:
            inferred = "jason_not_in_voice_labels"
            note = "Captions have speaker tags but none map to Jason Lemkin."
        return {
            "method": "vtt_voice_tags",
            "voice_tags_in_captions": True,
            "total_cues": len(cues),
            "cues_with_speaker_label": len(labeled),
            "cues_labeled_jason_lemkin": len(jason_labeled),
            "jason_labeled_fraction_of_labeled": round(frac_l, 4),
            "jason_labeled_fraction_of_all_cues": round(frac_a, 4),
            "unique_speaker_labels": labels,
            "inferred_primary_speaker": inferred,
            "confidence_note": note,
        }

    inf, note = heuristic()
    return {
        "method": "title_channel_heuristic",
        "voice_tags_in_captions": False,
        "total_cues": len(cues),
        "cues_with_speaker_label": 0,
        "cues_labeled_jason_lemkin": 0,
        "jason_labeled_fraction_of_labeled": None,
        "jason_labeled_fraction_of_all_cues": None,
        "unique_speaker_labels": [],
        "inferred_primary_speaker": inf,
        "confidence_note": note,
    }


def append_matched_videos_entry(path: str, entry: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


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
) -> Tuple[str, str, int, Optional[str], Optional[str], Optional[str], List[VttCue]]:
    """
    Download auto-captions with yt-dlp, parse WebVTT to plain text.
    Uses --write-info-json (not --dump-json) so subtitle files are still written to disk.

    Returns (language_code, transcript_text, cue_count, upload_date_yyyymmdd, upload_date_iso, description, cues).
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
        cue_objs = parse_vtt_cues(vtt_path)
        text = vtt_to_plain_text_from_cues(cue_objs)
        return base, text, len(cue_objs), upload_y, upload_iso, description, cue_objs
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
    parser.add_argument("--enable-title-filter", action="store_true", help="Filter by keyword titles (KEYWORD_RULES).")
    parser.add_argument(
        "--saasstr-jason-primary",
        action="store_true",
        help=(
            "SaaStr channel only: keep titles matching expanded Jason-primary heuristics "
            "(frameworks, AMA, Workshop Wednesday, keynotes, etc.). "
            "Does not filter the Jason M. Lemkin channel. Title-only; not perfect speaker detection."
        ),
    )
    parser.add_argument("--cookies", type=str, default=None, help="Optional Netscape cookies file for yt-dlp.")
    parser.add_argument("--proxy", type=str, default=None, help="Optional proxy URL for yt-dlp (--proxy).")
    parser.add_argument("--max-transcript-retries", type=int, default=2, help="Retries per video if yt-dlp fails.")
    parser.add_argument("--retry-backoff-seconds", type=float, default=3.0, help="Backoff between retries.")
    parser.add_argument("--sleep-seconds", type=float, default=2.0, help="Sleep after each video.")
    parser.add_argument(
        "--matched-videos-list",
        type=str,
        default=None,
        help=(
            "Append one JSON object per processed video (id, title, url, matched_keywords, "
            "transcript_ok, jason_speaker_attribution summary). Created next to output by default if path relative."
        ),
    )
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
    if args.saasstr_jason_primary and args.enable_title_filter:
        print(
            "Note: --saasstr-jason-primary takes precedence for SaaStr; "
            "Jason M. Lemkin channel is unfiltered. Other channels use --enable-title-filter."
        )

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
        list_url = resolve_uploads_playlist_url(ch.url)
        if list_url != ch.url:
            print(f"Using uploads playlist for full listing ({list_url})")
        videos = list_channel_videos(list_url, limit=list_limit)
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
            if args.saasstr_jason_primary and ch.name == "SaaStr":
                matched_keywords = match_saasstr_jason_primary(title)
                if not matched_keywords:
                    continue
            elif args.saasstr_jason_primary and ch.name == "Jason M. Lemkin":
                pass
            elif args.enable_title_filter:
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
                "jason_speaker_attribution": None,
                "transcript_text_jason_labeled_cues_only": None,
                "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
                "listing_source": list_url,
            }

            err: Optional[Exception] = None
            cue_objs: List[VttCue] = []
            for attempt in range(args.max_transcript_retries + 1):
                try:
                    lang, text, cue_count, upload_y, upload_iso, desc, cue_objs = fetch_transcript_ytdlp(
                        video_id,
                        preferred,
                        cookies_path=args.cookies,
                        proxy=args.proxy,
                    )
                    row["transcript_text"] = text
                    row["transcript_language"] = lang
                    row["transcript_segments_count"] = cue_count
                    row["upload_date"] = upload_y
                    row["upload_date_iso"] = upload_iso
                    row["description"] = desc
                    total_ok += 1
                    err = None
                    break
                except Exception as e:
                    err = e
                    cue_objs = []
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

            row["jason_speaker_attribution"] = build_jason_speaker_attribution(
                cue_objs,
                channel=ch.name,
                title=title,
                matched_keywords=matched_keywords,
                transcript_ok=err is None,
            )
            if err is None and cue_objs:
                jason_parts = [
                    c.text
                    for c in cue_objs
                    if c.speaker and _speaker_label_is_jason_lemkin(c.speaker)
                ]
                row["transcript_text_jason_labeled_cues_only"] = (
                    " ".join(jason_parts) if jason_parts else None
                )

            if args.matched_videos_list:
                ja = row["jason_speaker_attribution"]
                append_matched_videos_entry(
                    args.matched_videos_list,
                    {
                        "channel": ch.name,
                        "video_id": video_id,
                        "video_title": title,
                        "video_url": page_url,
                        "matched_keywords": matched_keywords,
                        "listing_source": list_url,
                        "transcript_ok": err is None,
                        "inferred_primary_speaker": ja.get("inferred_primary_speaker"),
                        "jason_attribution_method": ja.get("method"),
                        "voice_tags_in_captions": ja.get("voice_tags_in_captions"),
                        "cues_labeled_jason_lemkin": ja.get("cues_labeled_jason_lemkin"),
                        "cues_with_speaker_label": ja.get("cues_with_speaker_label"),
                        "total_cues": ja.get("total_cues"),
                        "confidence_note": ja.get("confidence_note"),
                    },
                )

            append_jsonl(args.output, row)
            seen.add(video_id)
            if args.resume:
                save_seen_video_ids(args.seen_cache, seen)
            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)

    print(f"\nDone. matched_videos={total_matched}, with_transcript_text={total_ok}")
    print(f"Dataset: {args.output}")
    if args.matched_videos_list:
        print(f"Matched videos list: {args.matched_videos_list}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
