from __future__ import annotations

import sys
from pathlib import Path

import hashlib
import html as html_lib
import json
import os
import re
import time
import warnings
from typing import Iterable
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

# Ensure imports work when running as a standalone script from repo root.
REPO_ROOT = str(Path(__file__).resolve().parents[2])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from persona_pipeline.config.config import (
    BLOG_BASE_URL,
    BLOG_DELAY_SECONDS,
    BLOG_LIMIT,
    BLOG_USER_AGENT,
    MIN_WORD_COUNT,
    PERSONA_USERNAME,
    persona_raw_path,
)


KALZUMEUS_URL_RE = re.compile(r"^https?://www\.kalzumeus\.com/20\d{2}/", re.IGNORECASE)


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def clean_text(text: str) -> str:
    """
    Basic text cleaning for Phase 1 filtering.

    Phase 2 Spark will do additional cleaning, but we still want stable word counts.
    """
    text = html_lib.unescape(text)
    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)
    # Strip common markdown artifacts that sometimes appear in scraped text
    text = re.sub(r"(\*\*|~~|`|##+|__)", " ", text)
    # Remove markdown links: [text](url) -> text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    return _normalize_whitespace(text)


def discover_essay_urls(session: requests.Session) -> list[str]:
    """
    Discover essay URLs by scanning Kalzumeus archive-like pages.
    """
    discovery_pages = [
        urljoin(BLOG_BASE_URL + "/", "greatest-hits/"),
        urljoin(BLOG_BASE_URL + "/", "archive/"),
        BLOG_BASE_URL + "/",
    ]

    urls: set[str] = set()
    headers = {"User-Agent": BLOG_USER_AGENT}

    for page_url in discovery_pages:
        try:
            resp = session.get(page_url, headers=headers, timeout=30)
            resp.raise_for_status()
        except Exception as e:
            warnings.warn(f"Blog discovery failed for {page_url}: {e}")
            continue

        soup = BeautifulSoup(resp.text, "html.parser")
        for a in soup.select("a[href]"):
            href = a.get("href")
            if not href:
                continue
            full_url = urljoin(BLOG_BASE_URL + "/", href)
            full_url = full_url.split("#", 1)[0]
            if KALZUMEUS_URL_RE.match(full_url):
                urls.add(full_url)

    return sorted(urls)


def extract_article_text(soup: BeautifulSoup) -> str:
    # Prefer the most specific containers first.
    container = (
        soup.find("article")
        or soup.select_one("div.entry-content")
        or soup.select_one("div.post-content")
    )
    if container is None:
        container = soup
    return container.get_text(" ", strip=True)


def url_to_stable_id(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]


def scrape_blog_urls(session: requests.Session, urls: Iterable[str]) -> list[dict]:
    out_items: list[dict] = []
    headers = {"User-Agent": BLOG_USER_AGENT}

    for i, url in enumerate(urls, start=1):
        try:
            resp = session.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            title_tag = soup.find("h1")
            title = title_tag.get_text(" ", strip=True) if title_tag else url

            raw_text = extract_article_text(soup)
            cleaned_text = clean_text(raw_text)
            word_count = len(cleaned_text.split())

            # Filter stub pages
            if word_count < 100:
                continue

            out_items.append(
                {
                    "id": url_to_stable_id(url),
                    "source": "blog",
                    "url": url,
                    "title": title,
                    "text": cleaned_text,
                    "word_count": word_count,
                    "author": PERSONA_USERNAME,
                }
            )
        except Exception as e:
            warnings.warn(f"Blog scrape failed for {url}: {e}")
        finally:
            # Polite delay between requests
            time.sleep(BLOG_DELAY_SECONDS)

        if len(out_items) >= BLOG_LIMIT:
            break

        if i % 10 == 0:
            print(f"[collect_blog] scraped {len(out_items)} items...")

    return out_items


def main() -> None:
    os.makedirs(os.path.dirname(persona_raw_path("x")), exist_ok=True)

    session = requests.Session()
    discovered = discover_essay_urls(session)
    if not discovered:
        warnings.warn("No blog URLs discovered; writing empty output.")
        with open(persona_raw_path("patio11_blog.json"), "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        return

    # Process only the first BLOG_LIMIT discovered URLs to keep local runs manageable.
    urls_to_process = discovered[:BLOG_LIMIT]
    print(f"[collect_blog] discovered {len(discovered)} URLs; processing {len(urls_to_process)}")

    items = scrape_blog_urls(session, urls_to_process)

    out_path = persona_raw_path("patio11_blog.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    print(f"[collect_blog] wrote {len(items)} items -> {out_path}")


if __name__ == "__main__":
    main()

