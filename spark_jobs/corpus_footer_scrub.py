"""
Strip SaaStr / blog recirculation tails and noisy social footers from corpus text.

Used by Spark (MinIO raw→processed), host_finetune cleaners, and RAG context
formatting so training + retrieval are not dominated by "Related Posts" blocks.

Kept under ``spark_jobs/`` so ``clean_and_embed.py`` can import it when Airflow
SparkSubmit runs from ``/opt/airflow/spark_jobs`` (that directory is volume-mounted;
repo root is not).
"""
from __future__ import annotations

import re

# Trailing blocks (WordPress / SaaStr style).
_FOOTER_START_PATTERNS = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\n\s*Related Posts\s*\n",
        r"\n\s*Related articles\s*\n",
        r"\n\s*Related Article\s*\n",
        r"\n\s*You May Also Like\s*\n",
        r"\n\s*READ MORE\s*\n",
        r"\n\s*Read more on SaaStr\s*\n",
        r"\n\s*Continue reading\s*\n",
    )
)

# Tweet / X attribution tail often pasted into exports.
_TWITTER_TAIL = re.compile(
    r"\n\s*[—–-]\s*Jason\b.*\Z",
    re.DOTALL | re.IGNORECASE,
)


def strip_footer_noise(text: str, *, footer_zone_frac: float = 0.62) -> str:
    """Remove recirculation footer and common social tails.

    ``footer_zone_frac``: only treat a match as a footer if it starts at or
    after this fraction of string length (avoids killing mid-article mentions).
    """
    if not text:
        return text
    s = text.strip()
    if len(s) < 120:
        return text

    cut = len(s)
    zone = int(len(s) * footer_zone_frac)

    for rx in _FOOTER_START_PATTERNS:
        for m in rx.finditer(s):
            if m.start() >= zone:
                cut = min(cut, m.start())

    if cut < len(s):
        s = s[:cut].rstrip()

    s = _TWITTER_TAIL.sub("", s).rstrip()
    return s
