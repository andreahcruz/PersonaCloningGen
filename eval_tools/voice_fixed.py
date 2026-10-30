"""Voice score on a fixed number of words, so draft length cannot drive it (PUK-28).

The style distance averages rates per 100 words, so a longer text lands closer to the Lemkin
profile just for being longer (log word count and distance correlate at about -0.66 over real
posts and drafts, see ``eval_tools.length_control``). Two runs that write different lengths
therefore differ in "voice" partly because of length. Kevin chose a fixed-length score as the fix.

This tool scores **the first N words** of every draft with Kevin's own style distance, so every
draft is measured on the same amount of text:

* blog (the medium where the style distance is a valid selector) and LinkedIn (diagnostic) use
  N = 100 by default;
* a draft with fewer than N words is **not scored** (status ``too_short``). Dropping short drafts
  is the cost of this method, so the summary says how many were dropped. A run that writes many
  short drafts therefore has fewer scored drafts, and ``compare`` only uses drafts that both runs
  have a score for;
* X and talk are **not supported**. Relabel's X posts have a median of 16 words (PersonaRAG's is
  126), too few for any fixed length, and talk has no reference posts.

    python -m eval_tools.voice_fixed --kevin-root ../kevin-ws --traces experiments/EXP-.../raw/relabel.jsonl \
        --out-jsonl voice_fixed_rows.jsonl --out-json voice_fixed_summary.json

``--out-jsonl`` can be passed to ``eval_tools.compare`` with ``--extra-a`` or ``--extra-b`` and
compared as ``style_distance_fixed`` (lower is closer to Lemkin).

It needs Kevin's code and reference data, which are on branch ``kevin/298b-integration``. Do not clone
that branch normally (about 480 MB of adapter files in Git LFS); skip them:

    GIT_LFS_SKIP_SMUDGE=1 git worktree add ../kevin-ws origin/kevin/298b-integration

Choosing N is a trade-off: a larger N is a fairer length control but drops more short drafts. Run it
with a few values (100 and 150 are sensible for blogs) and check that the conclusion does not change.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable, Mapping

from eval_tools.length_control import load_kevin_distance, truncate_words, word_count

FIXED_WORDS = {"blog": 100, "linkedin": 100}


def fixed_length_distance(text: str, n: int, distance: Callable[[str], float | None]) -> tuple[float | None, str]:
    """Style distance of the first ``n`` words, with a status: scored, too_short or no_distance."""
    if word_count(text) < n:
        return None, "too_short"
    value = distance(truncate_words(text, n))
    return (None, "no_distance") if value is None else (round(value, 6), "scored")


def score_drafts(
    rows: Iterable[dict],
    distances: Mapping[str, Callable[[str], float | None]],
    fixed_words: Mapping[str, int] = FIXED_WORDS,
    text_field: str = "answer",
) -> list[dict]:
    """One result per draft. ``style_distance_fixed`` is None unless the draft was scored."""
    results = []
    for row in rows:
        medium = str(row.get("medium"))
        text = row.get(text_field) or ""
        item = {
            "id": str(row.get("id")),
            "medium": medium,
            "voice_words": word_count(text),
            "voice_fixed_words": fixed_words.get(medium),
            "style_distance_fixed": None,
        }
        if medium not in fixed_words or medium not in distances:
            item["voice_fixed_status"] = "not_supported"
        else:
            item["style_distance_fixed"], item["voice_fixed_status"] = fixed_length_distance(
                text, fixed_words[medium], distances[medium]
            )
        results.append(item)
    return results


def summarize(results: list[dict], real: Mapping[str, dict] | None = None) -> dict:
    """Per medium: how many drafts were scored or dropped, and the mean fixed-length distance."""
    summary: dict = {"drafts": {}, "real_posts": dict(real or {})}
    for medium in sorted({r["medium"] for r in results}):
        items = [r for r in results if r["medium"] == medium]
        status = Counter(r["voice_fixed_status"] for r in items)
        scored = [r["style_distance_fixed"] for r in items if r["style_distance_fixed"] is not None]
        summary["drafts"][medium] = {
            "n_drafts": len(items),
            "fixed_words": items[0]["voice_fixed_words"],
            "scored": status.get("scored", 0),
            "too_short": status.get("too_short", 0),
            "not_supported": status.get("not_supported", 0),
            "share_scored": round(status.get("scored", 0) / len(items), 4),
            "mean_distance_fixed": round(statistics.fmean(scored), 4) if scored else None,
            "median_words": statistics.median(r["voice_words"] for r in items),
        }
    return summary


def real_post_reference(real_texts: Iterable[str], n: int, distance: Callable[[str], float | None]) -> dict:
    """Context for the numbers: how real Lemkin posts score on the same first N words."""
    texts = list(real_texts)
    long_enough = [t for t in texts if word_count(t) >= n]
    values = [d for d in (distance(truncate_words(t, n)) for t in long_enough) if d is not None]
    return {
        "n_real_posts": len(texts),
        "n_at_least_n_words": len(long_enough),
        "fixed_words": n,
        "mean_distance_fixed": round(statistics.fmean(values), 4) if values else None,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kevin-root", required=True, help="Folder with Kevin's host_finetune/ and experiments/ data.")
    p.add_argument("--traces", required=True, help="JSONL of drafts with id, medium and the text field.")
    p.add_argument("--text-field", default="answer")
    p.add_argument("--blog-words", type=int, default=FIXED_WORDS["blog"])
    p.add_argument("--linkedin-words", type=int, default=FIXED_WORDS["linkedin"])
    p.add_argument("--out-jsonl", help="Per draft results, readable by eval_tools.compare --extra-a/--extra-b.")
    p.add_argument("--out-json", help="Summary per medium.")
    args = p.parse_args()

    fixed = {"blog": args.blog_words, "linkedin": args.linkedin_words}
    distances, real = {}, {}
    for medium in fixed:
        distance, real_texts = load_kevin_distance(args.kevin_root, medium)
        distances[medium] = distance
        real[medium] = real_post_reference(real_texts, fixed[medium], distance)
    with open(args.traces, encoding="utf-8-sig") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    results = score_drafts(rows, distances, fixed, args.text_field)
    summary = summarize(results, real)
    if args.out_jsonl:
        with open(args.out_jsonl, "w", encoding="utf-8", newline="\n") as out:
            for item in results:
                out.write(json.dumps(item, ensure_ascii=False) + "\n")
    if args.out_json:
        Path(args.out_json).write_bytes(json.dumps(summary, indent=2).encode("utf-8"))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
