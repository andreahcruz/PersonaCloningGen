"""Cheap sanity checks on model output (no third-party deps).

Lives under ``spark_jobs/`` next to ``corpus_footer_scrub.py`` so the Streamlit app,
``generate.py`` and the host-side fine-tune scripts can all import it the same way.

The failure this guards against is real: a bad GGUF export once produced pure
garbage ("vibe vibe vibe ...") while training loss looked normal (see
``host_finetune/merge_and_export.py``).
"""
from __future__ import annotations

import re
from collections import Counter

_WORD_RE = re.compile(r"\w+")


def looks_degenerate(
    text: str,
    *,
    min_words: int = 40,
    window_words: int = 400,
    max_top_share: float = 0.35,
    min_unique_ratio: float = 0.15,
) -> bool:
    """True if ``text`` looks like a collapsed model (one token repeated over and over).

    Only the first ``window_words`` words are inspected so the unique-word ratio does
    not shrink just because a legitimate draft is long. Short outputs are never flagged.
    Normal English prose has a most-common-word share around 5-8% and a unique ratio
    above 0.3, so the defaults leave a wide margin.
    """
    words = _WORD_RE.findall((text or "").lower())[:window_words]
    if len(words) < min_words:
        return False
    counts = Counter(words)
    top_share = counts.most_common(1)[0][1] / len(words)
    unique_ratio = len(counts) / len(words)
    return top_share > max_top_share or unique_ratio < min_unique_ratio
