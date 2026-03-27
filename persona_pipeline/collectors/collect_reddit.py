from __future__ import annotations

import sys
from pathlib import Path

import html as html_lib
import json
import os
import re
import time
import warnings
from typing import Any, Optional

import requests

# Ensure imports work when running as a standalone script from repo root.
REPO_ROOT = str(Path(__file__).resolve().parents[2])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from persona_pipeline.config.config import (
    REDDIT_LIMIT,
    REDDIT_PAGE_DELAY_SECONDS,
    REDDIT_USER_AGENT,
    REDDIT_429_RETRY_SECONDS,
    PERSONA_USERNAME,
    persona_raw_path,
)


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def clean_text(text: str) -> str:
    """
    Basic cleanup for Reddit fields.

    Reddit often returns markdown-like text; Spark Phase 2 will do richer cleaning.
    """
    text = html_lib.unescape(text or "")
    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)
    # Remove markdown links: [text](url) -> text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Strip common markdown symbols
    text = re.sub(r"(\*\*|__|~~|`|#+|>+)", " ", text)
    return _normalize_whitespace(text)


def reddit_url(endpoint: str) -> str:
    return f"https://www.reddit.com{endpoint}"


def fetch_json_with_429(session: requests.Session, url: str, timeout_s: int = 30) -> Any:
    """
    Retry on HTTP 429 (rate limited) as required by the brief.
    """
    while True:
        resp = session.get(url, timeout=timeout_s)
        if resp.status_code == 429:
            warnings.warn(f"Reddit 429 rate limit hit; sleeping {REDDIT_429_RETRY_SECONDS}s then retrying.")
            time.sleep(REDDIT_429_RETRY_SECONDS)
            continue
        resp.raise_for_status()
        return resp.json()


def iter_reddit_items(
    session: requests.Session,
    endpoint: str,
    max_items: int,
    item_kind: str,
) -> list[dict]:
    """
    Fetch items from a reddit user JSON endpoint using pagination tokens.
    """
    out: list[dict] = []
    after: Optional[str] = None

    while len(out) < max_items:
        base = reddit_url(endpoint)
        # Compose ?limit=100 and optional after=
        if after:
            url = f"{base}?limit=100&after={after}"
        else:
            url = f"{base}?limit=100"

        try:
            payload = fetch_json_with_429(session, url)
        except Exception as e:
            warnings.warn(f"Reddit page fetch failed for {url}: {e}")
            break

        data = payload.get("data") or {}
        children = data.get("children") or []
        after = data.get("after")

        for child in children:
            item = child.get("data") or {}
            subreddit = item.get("subreddit")
            post_id = item.get("id")
            author = item.get("author")
            score = item.get("score")
            permalink = item.get("permalink")
            url = f"https://www.reddit.com{permalink}" if permalink else ""

            if item_kind == "comments":
                body = item.get("body") or ""
                if body in ("[deleted]", "[removed]"):
                    continue
                text = clean_text(body)
                word_count = len(text.split())
                out.append(
                    {
                        "id": post_id,
                        "source": "reddit_comment",
                        "subreddit": subreddit,
                        "url": url,
                        "text": text,
                        "word_count": word_count,
                        "score": score,
                        "author": author or PERSONA_USERNAME,
                    }
                )
            elif item_kind == "posts":
                # Only keep self-posts (is_self == True)
                if not item.get("is_self"):
                    continue
                title = item.get("title") or ""
                selftext = item.get("selftext") or ""
                if selftext in ("[deleted]", "[removed]"):
                    continue
                full_text = clean_text(f"{title}\n{selftext}".strip())
                word_count = len(full_text.split())
                out.append(
                    {
                        "id": post_id,
                        # Keep the same source label as comments to match later pipeline prompts/logic.
                        "source": "reddit_comment",
                        "subreddit": subreddit,
                        "url": url,
                        "text": full_text,
                        "word_count": word_count,
                        "score": score,
                        "author": author or PERSONA_USERNAME,
                    }
                )
            else:
                raise ValueError(f"Unknown item_kind: {item_kind}")

            if len(out) >= max_items:
                break

        # Delay between page requests (polite crawling).
        time.sleep(REDDIT_PAGE_DELAY_SECONDS)

        if not after:
            break

    return out[:max_items]


def main() -> None:
    os.makedirs(os.path.dirname(persona_raw_path("x")), exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": REDDIT_USER_AGENT})

    # Collect comments and self-posts; cap total items at REDDIT_LIMIT.
    items: list[dict] = []

    try:
        comments = iter_reddit_items(
            session=session,
            endpoint="/user/patio11/comments.json",
            max_items=REDDIT_LIMIT,
            item_kind="comments",
        )
        items.extend(comments)

        remaining = max(0, REDDIT_LIMIT - len(items))
        if remaining > 0:
            posts = iter_reddit_items(
                session=session,
                endpoint="/user/patio11/submitted.json",
                max_items=remaining,
                item_kind="posts",
            )
            items.extend(posts)
    except Exception as e:
        warnings.warn(f"Reddit collection failed: {e}")

    out_path = persona_raw_path("patio11_reddit.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    print(f"[collect_reddit] wrote {len(items)} items -> {out_path}")


if __name__ == "__main__":
    main()

