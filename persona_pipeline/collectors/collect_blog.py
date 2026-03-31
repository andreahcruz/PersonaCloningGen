"""
Collect Jason Lemkin's blog posts from SaaStr.

Paginates https://www.saastr.com/author/jasonlkn/page/{n}/ and extracts
title, content, and date from each post page.

Run: python persona_pipeline/collectors/collect_blog.py
"""
from __future__ import annotations

import hashlib
import html as html_lib
import json
import os
import re
import sys
import time
import warnings
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

REPO_ROOT = str(Path(__file__).resolve().parents[2])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from persona_pipeline.config.config import (
    BLOG_AUTHOR_SLUG,
    BLOG_BASE_URL,
    BLOG_DELAY_SECONDS,
    BLOG_LIMIT,
    BLOG_USER_AGENT,
    MIN_WORD_COUNT,
    PERSONA_USERNAME,
    persona_raw_path,
)


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def clean_text(text: str) -> str:
    text = html_lib.unescape(text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"(\*\*|~~|`|##+|__)", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    return _normalize_whitespace(text)


def url_to_stable_id(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]


def discover_post_urls(session: requests.Session) -> list[str]:
    """Paginate the SaaStr author archive to find post URLs."""
    author_base = f"{BLOG_BASE_URL}/author/{BLOG_AUTHOR_SLUG}"
    headers = {"User-Agent": BLOG_USER_AGENT}
    urls: list[str] = []
    seen: set[str] = set()

    for page_num in range(1, 500):
        page_url = f"{author_base}/" if page_num == 1 else f"{author_base}/page/{page_num}/"
        print(f"[collect_blog] scanning page {page_num}: {page_url}")

        try:
            resp = session.get(page_url, headers=headers, timeout=30)
            if resp.status_code == 404:
                print(f"[collect_blog] page {page_num} returned 404; end of archive.")
                break
            resp.raise_for_status()
        except Exception as e:
            warnings.warn(f"Blog page fetch failed for {page_url}: {e}")
            break

        soup = BeautifulSoup(resp.text, "html.parser")
        cards = soup.select("h2.entry-title a[href]")
        if not cards:
            print("[collect_blog] no post links found on page; end of archive.")
            break

        for a in cards:
            href = a.get("href")
            if not href:
                continue
            full_url = urljoin(page_url, href).split("#", 1)[0]
            if full_url not in seen:
                seen.add(full_url)
                urls.append(full_url)

        time.sleep(BLOG_DELAY_SECONDS)

        if len(urls) >= BLOG_LIMIT:
            break

    return urls[:BLOG_LIMIT]


def extract_article_text(soup: BeautifulSoup) -> str:
    container = soup.select_one("div.entry-content")
    if container is None:
        container = soup.find("article") or soup
    return container.get_text(" ", strip=True)


def scrape_blog_urls(session: requests.Session, urls: list[str]) -> list[dict]:
    out_items: list[dict] = []
    headers = {"User-Agent": BLOG_USER_AGENT}

    for i, url in enumerate(urls, start=1):
        try:
            resp = session.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            title_tag = soup.select_one("h1.entry-title") or soup.find("h1")
            title = title_tag.get_text(" ", strip=True) if title_tag else url

            raw_text = extract_article_text(soup)
            cleaned_text = clean_text(raw_text)
            word_count = len(cleaned_text.split())

            if word_count < MIN_WORD_COUNT:
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
            time.sleep(BLOG_DELAY_SECONDS)

        if len(out_items) >= BLOG_LIMIT:
            break

        if i % 10 == 0:
            print(f"[collect_blog] scraped {len(out_items)} items...")

    return out_items


def main() -> None:
    os.makedirs(os.path.dirname(persona_raw_path("x")), exist_ok=True)

    session = requests.Session()
    discovered = discover_post_urls(session)
    if not discovered:
        warnings.warn("No blog URLs discovered; writing empty output.")
        with open(persona_raw_path("lemkin_blog.json"), "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        return

    print(f"[collect_blog] discovered {len(discovered)} URLs; scraping content")

    items = scrape_blog_urls(session, discovered)

    out_path = persona_raw_path("lemkin_blog.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    print(f"[collect_blog] wrote {len(items)} items -> {out_path}")


if __name__ == "__main__":
    main()
