from __future__ import annotations

import sys
from pathlib import Path

import html as html_lib
import json
import os
import re
import time
import warnings
from typing import Any

import requests
from bs4 import BeautifulSoup

# Ensure imports work when running as a standalone script from repo root.
REPO_ROOT = str(Path(__file__).resolve().parents[2])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from persona_pipeline.config.config import (
    HN_LIMIT,
    HN_ITEM_DELAY_SECONDS,
    PERSONA_USERNAME,
    HN_USERNAME,
    persona_raw_path,
)


BASE_URL = "https://hacker-news.firebaseio.com/v0"


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def clean_text_from_html(raw_html: str) -> str:
    """
    Convert HTML-ish HN fields into plain text suitable for word counting.
    """
    raw_html = html_lib.unescape(raw_html or "")
    soup = BeautifulSoup(raw_html, "html.parser")
    text = soup.get_text(" ", strip=True)
    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)
    # Strip markdown-like artifacts (rare in HN but harmless)
    text = re.sub(r"(\*\*|~~|`|##+|__)", " ", text)
    text = _normalize_whitespace(text)
    return text


def fetch_json(session: requests.Session, url: str, timeout_s: int = 30) -> Any:
    resp = session.get(url, timeout=timeout_s)
    resp.raise_for_status()
    return resp.json()


def collect_hn_items() -> list[dict]:
    session = requests.Session()
    items_out: list[dict] = []

    user_url = f"{BASE_URL}/user/{HN_USERNAME}.json"
    try:
        user_obj = fetch_json(session, user_url)
    except Exception as e:
        warnings.warn(f"HackerNews user fetch failed ({user_url}): {e}")
        return items_out

    submitted = user_obj.get("submitted") or []
    if not isinstance(submitted, list):
        warnings.warn("HN user returned unexpected 'submitted' shape; writing empty output.")
        return items_out

    for idx, item_id in enumerate(submitted[:HN_LIMIT], start=1):
        item_url = f"{BASE_URL}/item/{item_id}.json"
        try:
            item_obj = fetch_json(session, item_url)
        except Exception as e:
            warnings.warn(f"HN item fetch failed for id={item_id}: {e}")
            time.sleep(HN_ITEM_DELAY_SECONDS)
            continue

        try:
            item_type = item_obj.get("type")
            author = item_obj.get("by")
            score = item_obj.get("score")
            t = item_obj.get("time")
            url = f"https://news.ycombinator.com/item?id={item_id}"

            text = ""
            if item_type == "comment":
                raw_text = item_obj.get("text")
                if not raw_text:
                    continue
                text = clean_text_from_html(str(raw_text))
            elif item_type == "story":
                title = item_obj.get("title") or ""
                raw_body = item_obj.get("text") or ""
                body_text = clean_text_from_html(str(raw_body)) if raw_body else ""
                text = _normalize_whitespace(f"{title}\n{body_text}".strip())
            else:
                continue

            if not text:
                continue

            word_count = len(text.split())
            # Keep only meaningful texts (per brief: 30+ words).
            if word_count < 30:
                continue

            items_out.append(
                {
                    "id": item_id,
                    "source": "hackernews",
                    "url": url,
                    "text": text,
                    "word_count": word_count,
                    "score": score,
                    "time": t,
                    "author": author or PERSONA_USERNAME,
                }
            )
        except Exception as e:
            warnings.warn(f"HN item parse failed for id={item_id}: {e}")
        finally:
            # Polite delay between item fetches
            time.sleep(HN_ITEM_DELAY_SECONDS)

        if idx % 50 == 0:
            print(f"[collect_hn] processed {idx} item ids; collected {len(items_out)} texts...")

    return items_out


def main() -> None:
    os.makedirs(os.path.dirname(persona_raw_path("x")), exist_ok=True)
    out_path = persona_raw_path("patio11_hn.json")
    items = collect_hn_items()

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    print(f"[collect_hn] wrote {len(items)} items -> {out_path}")


if __name__ == "__main__":
    main()

