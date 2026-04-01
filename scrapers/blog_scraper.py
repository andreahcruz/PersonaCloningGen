import requests
from bs4 import BeautifulSoup
import json
import time
from urllib.parse import urljoin


def _parse_post_page(post_url: str, session: requests.Session) -> dict | None:
    """Extract title/date/content/tags from a SaaStr post page."""
    resp = session.get(post_url, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # Title
    title_node = soup.select_one("h1.entry-title")
    title = title_node.get_text(strip=True) if title_node else None

    # Date
    date_iso = None

    # Common WordPress-style meta tags.
    meta_dt = soup.select_one('meta[property="article:published_time"], meta[name="article:published_time"], meta[name="pubdate"]')
    if meta_dt and meta_dt.get("content"):
        date_iso = meta_dt.get("content")

    # Some templates use <time datetime="...">.
    if not date_iso:
        time_node = soup.select_one("time[datetime]")
        if time_node:
            date_iso = time_node.get("datetime")

    # Fallback: grab the rendered date string.
    if not date_iso:
        time_node = soup.select_one("time")
        if time_node:
            date_iso = time_node.get_text(strip=True) or None

    # Content
    content_node = soup.select_one("div.entry-content")
    content = content_node.get_text(separator="\n", strip=True) if content_node else None

    # Tags (optional)
    tags = []
    for tag_a in soup.select('a[rel="tag"]'):
        t = tag_a.get_text(strip=True)
        if t:
            tags.append(t)

    if not title or not content:
        return None

    return {
        "title": title,
        "date": date_iso,
        "url": post_url,
        "tags": tags,
        "content": content,
    }


def scrape_saastr_author(author_slug: str, output_path: str, *, delay_s: float = 1.0, max_pages: int = 200):
    """
    Scrape all posts for a SaaStr author, following /author/<slug>/page/N/ until no posts are found.

    Writes one JSON record per line to `output_path`.
    """
    author_base = f"https://www.saastr.com/author/{author_slug}"
    headers = {
        # Identify ourselves politely. SaaStr may rate-limit default tooling UAs.
        "User-Agent": "Mozilla/5.0 (compatible; DataScrapingBot/1.0; +https://example.com)",
        "Accept-Language": "en-US,en;q=0.9",
    }

    session = requests.Session()
    session.headers.update(headers)

    seen_urls = set()

    # Stream writing so we don't lose work if the script stops.
    with open(output_path, "w", encoding="utf-8") as f:
        for p in range(1, max_pages + 1):
            page_url = author_base + "/" if p == 1 else f"{author_base}/page/{p}/"
            print(f"Scraping author page {p}: {page_url}")

            resp = session.get(page_url, timeout=30)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            # Listing cards on SaaStr author pages are typically <h2 class="entry-title"><a href="...">...</a></h2>
            cards = soup.select("h2.entry-title a[href]")
            if not cards:
                print("No posts found; assuming we've reached the end.")
                break

            post_urls = []
            for a in cards:
                href = a.get("href")
                if not href:
                    continue
                post_urls.append(urljoin(page_url, href))

            # De-dupe within and across pages.
            for post_url in sorted(set(post_urls)):
                if post_url in seen_urls:
                    continue
                seen_urls.add(post_url)

                print(f"  Fetching post: {post_url}")
                try:
                    post_data = _parse_post_page(post_url, session)
                    if not post_data:
                        print("    Skipped (missing title/content).")
                        continue
                    post_data["scraped_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                    f.write(json.dumps(post_data, ensure_ascii=False) + "\n")
                    f.flush()
                except Exception as e:
                    print(f"    Error scraping post: {e}")

                time.sleep(delay_s)  # Be polite to the server


if __name__ == "__main__":
    # Change output file name if you want.
    scrape_saastr_author(
        author_slug="jasonlkn",
        output_path="lemkin_blog.jsonl",
        delay_s=1.0,
        max_pages=20000,
    )