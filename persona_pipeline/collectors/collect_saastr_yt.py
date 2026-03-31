"""
Collect YouTube transcripts from the SaaStr channel (@Saastr).

Uses yt-dlp to download auto-generated captions and converts them to plain text.

Run: python persona_pipeline/collectors/collect_saastr_yt.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = str(Path(__file__).resolve().parents[2])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from persona_pipeline.config.config import SAASTR_YT_CHANNEL, YT_MAX_VIDEOS
from persona_pipeline.collectors._yt_common import collect_channel


def main() -> None:
    collect_channel(
        channel_url=SAASTR_YT_CHANNEL,
        source_label="saastr_video",
        output_filename="saastr_yt.json",
        limit=YT_MAX_VIDEOS,
    )


if __name__ == "__main__":
    main()
