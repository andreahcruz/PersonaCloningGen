"""What did adding RAG change, and what did it cost? (PUK-34)

``eval_tools.compare`` answers "did quality change?". This tool answers the other half: what
retrieval did to the drafts and what it cost, using the fields saved with each draft.

Both runs (paired by topic and medium):
    prompt tokens, generated tokens, words, how often the draft ended by hitting the length
    limit, and how often it ended on its own while using under half the token budget (Kevin's
    "early_eos"; it is not a failure, just a short draft).

Retrieval run only (these only exist when retrieval was on):
    extra seconds spent extracting evidence (counted once per topic, because the same evidence
    is shared by the four mediums of a topic), drafts with unsupported numbers, drafts flagged for
    copying a retrieved passage, how many proposed claims survived verification, and how well the
    retrieved passages fit the requested medium.

    python -m eval_tools.rag_effect report --a-drafts off_raw.jsonl --b-drafts on_raw.jsonl \
        --label-a "RAG off" --label-b "RAG on" --out-md effect.md

The raw draft files are large (4 MB for one run), so ``slim`` keeps only the fields this tool reads:

    python -m eval_tools.rag_effect slim --raw on_raw.jsonl --out on_slim.jsonl

``report`` accepts raw or slim files. Generation time is not saved with the drafts, so cost is
measured in tokens and in the extra evidence seconds, not in total latency.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Iterable

from eval_tools.bootstrap_ci import summarize_paired

Key = tuple[str, str]
MEDIUMS = ("blog", "linkedin", "x", "talk")
# Passage mediums that fit each requested medium. Talk is matched by transcripts only.
FITS = {"blog": {"blog"}, "linkedin": {"linkedin"}, "x": {"x"}, "talk": {"youtube_jason", "youtube_saastr", "talk"}}

PAIRED_METRICS = {
    "prompt_tokens": "Prompt tokens",
    "generated_tokens": "Generated tokens",
    "answer_words": "Words in the draft",
    "stopped_at_length": "Ended at the length limit (share)",
    "early_eos": "Ended on its own using under half the token budget (share)",
}


def slim_draft(row: dict) -> dict:
    """Keep only what ``report`` reads. Works for runs with and without retrieval."""
    retrieval = bool(row.get("retrieval"))
    numeric = row.get("numeric_grounding") or {}
    copying = row.get("rag_source_copying") or {}
    hits = row.get("retrieval_hits") or []
    out = {
        "id": str(row.get("id")),
        "medium": str(row.get("medium")),
        "retrieval": retrieval,
        "prompt_tokens": row.get("prompt_tokens"),
        "generated_tokens": row.get("generated_tokens_including_stop", row.get("generated_tokens")),
        "answer_words": row.get("answer_words", len((row.get("answer") or "").split())),
        "stop_reason": row.get("stop_reason"),
        "early_eos": row.get("early_eos"),
    }
    if retrieval:
        out.update(
            {
                "extraction_seconds": row.get("extraction_seconds"),
                "verification_seconds": row.get("verification_seconds"),
                # A slim row has these already flattened. Read them back instead of dropping them.
                "numeric_applicable": numeric.get("applicable", row.get("numeric_applicable")),
                "numeric_pass": numeric.get("pass", row.get("numeric_pass")),
                "copy_flagged": copying.get("flagged", row.get("copy_flagged")),
                "n_hits": len(hits) if hits else row.get("n_hits", 0),
                "hit_mediums": [h.get("medium") for h in hits] if hits else row.get("hit_mediums", []),
                "hit_distances": [h.get("distance") for h in hits] if hits else row.get("hit_distances", []),
                "n_proposed_claims": len(row.get("proposed_atomic_claims") or []) if "proposed_atomic_claims" in row else row.get("n_proposed_claims"),
                "n_verified_claims": len(row.get("verified_claims") or []) if "verified_claims" in row else row.get("n_verified_claims"),
                "accepted_attempt": row.get("accepted_attempt"),
                "repair_occurred": row.get("repair_occurred"),
            }
        )
    return out


def load_drafts(path: str | Path) -> dict[Key, dict]:
    out: dict[Key, dict] = {}
    with open(path, encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            slim = slim_draft(json.loads(line))
            key = (slim["id"], slim["medium"])
            if key in out:
                raise ValueError(f"Two drafts share the key {key} in {path}.")
            out[key] = slim
    return out


def _value(row: dict, metric: str) -> float | None:
    if metric == "stopped_at_length":
        reason = row.get("stop_reason")
        return None if reason is None else float(reason == "length")
    if metric == "early_eos":
        v = row.get("early_eos")
        return None if v is None else float(bool(v))
    v = row.get(metric)
    return None if v is None else float(v)


def paired_cost(a: dict[Key, dict], b: dict[Key, dict], *, n_boot: int = 5000, seed: int = 0) -> list[dict]:
    """Paired B minus A for the metrics both runs record, overall and per medium."""
    keys = sorted(set(a) & set(b))
    results = []
    for metric, title in PAIRED_METRICS.items():
        for scope in ("all", *MEDIUMS):
            pairs = []
            for k in keys:
                if scope != "all" and k[1] != scope:
                    continue
                va, vb = _value(a[k], metric), _value(b[k], metric)
                if va is not None and vb is not None:
                    pairs.append((k, va, vb))
            if not pairs:
                continue
            s = summarize_paired([y - x for _, x, y in pairs], [k[0] for k, _, _ in pairs], n_boot=n_boot, n_perm=n_boot, seed=seed)
            s.update(metric=metric, title=title, scope=scope,
                     mean_a=round(statistics.fmean(x for _, x, _ in pairs), 3), mean_b=round(statistics.fmean(y for _, _, y in pairs), 3))
            results.append(s)
    return results


def _rate(flags: list[bool | None]) -> float | None:
    known = [f for f in flags if f is not None]
    return round(sum(1 for f in known if f) / len(known), 4) if known else None


def rag_only(drafts: dict[Key, dict]) -> dict:
    """What retrieval did, from the retrieval run alone. Empty if that run had no retrieval."""
    rows = [r for r in drafts.values() if r.get("retrieval")]
    if not rows:
        return {"n": 0}

    def block(items: list[dict], overall: bool = False) -> dict:
        # Evidence is extracted once per topic and the same time is copied onto all four mediums of that topic.
        # Count it once per topic, or the total is four times too high.
        per_topic: dict[str, float] = {}
        for r in items:
            if r.get("extraction_seconds") is not None:
                per_topic.setdefault(r["id"], r["extraction_seconds"] + (r.get("verification_seconds") or 0))
        secs = list(per_topic.values())
        proposed = sum(r["n_proposed_claims"] or 0 for r in items)
        verified = sum(r["n_verified_claims"] or 0 for r in items)
        slots = fits = 0
        distances: list[float] = []
        for r in items:
            allowed = FITS.get(r["medium"], {r["medium"]})
            for m in r.get("hit_mediums") or []:
                slots += 1
                fits += m in allowed
            distances += [d for d in (r.get("hit_distances") or []) if d is not None]
        applicable = [r for r in items if r.get("numeric_applicable")]
        return {
            "n": len(items),
            "topics": len(secs),
            "evidence_seconds_per_topic_mean": round(statistics.fmean(secs), 1) if secs else None,
            "evidence_seconds_per_topic_median": round(statistics.median(secs), 1) if secs else None,
            # Totals only make sense for the whole run: per medium they would count the same topics again.
            "evidence_seconds_total": round(sum(secs), 0) if secs and overall else None,
            "evidence_seconds_per_draft": round(sum(secs) / len(items), 1) if secs and overall else None,
            "unsupported_numbers_rate": _rate([r.get("numeric_pass") is False for r in applicable]) if applicable else None,
            "unsupported_numbers_count": sum(1 for r in applicable if r.get("numeric_pass") is False),
            "copy_flag_rate": _rate([r.get("copy_flagged") for r in items]),
            "copy_flag_count": sum(1 for r in items if r.get("copy_flagged")),
            "claims_proposed": proposed,
            "claims_verified": verified,
            "claim_acceptance": round(verified / proposed, 4) if proposed else None,
            "second_attempt_rate": _rate([(r.get("accepted_attempt") or 1) > 1 for r in items]),
            "passage_fits_medium": round(fits / slots, 4) if slots else None,
            "mean_passage_distance": round(statistics.fmean(distances), 4) if distances else None,
        }

    out = {"overall": block(rows, overall=True), "by_medium": {}}
    for medium in MEDIUMS:
        items = [r for r in rows if r["medium"] == medium]
        if items:
            out["by_medium"][medium] = block(items)
    out["n"] = len(rows)
    return out


def _fmt(v, digits: int = 3) -> str:
    return "n/a" if v is None else f"{v:.{digits}f}"


def _pct(v) -> str:
    return "n/a" if v is None else f"{v:.0%}"


def to_markdown(cost: list[dict], effect: dict, label_a: str, label_b: str) -> str:
    lines = [
        f"# What {label_b} changed and what it cost, compared with {label_a}",
        "",
        f"Differences are {label_b} minus {label_a}, paired by topic and medium, with a 95 percent interval resampled by topic. "
        "Generation time is not saved with the drafts, so cost is in tokens and in the extra evidence seconds.",
        "",
        "## Both runs",
        "",
        f"| Measure | Scope | n | {label_a} | {label_b} | Difference | 95% interval |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in cost:
        interval = "n/a" if s["ci_low"] is None else f"{s['ci_low']:+.2f} to {s['ci_high']:+.2f}"
        lines.append(f"| {s['title']} | {s['scope']} | {s['n']} | {_fmt(s['mean_a'], 2)} | {_fmt(s['mean_b'], 2)} | {s['mean_diff']:+.2f} | {interval} |")
    if effect.get("n"):
        o = effect["overall"]
        lines += [
            "",
            f"## {label_b} only (retrieval on)",
            "",
            "| Measure | All drafts | " + " | ".join(m for m in MEDIUMS if m in effect["by_medium"]) + " |",
            "|---|---|" + "---|" * len(effect["by_medium"]),
        ]
        rows = [
            ("Drafts", "n", lambda v: str(v)),
            ("Topics", "topics", lambda v: str(v)),
            ("Evidence extraction seconds per topic (mean)", "evidence_seconds_per_topic_mean", lambda v: _fmt(v, 1)),
            ("Total extra evidence seconds (counted once per topic)", "evidence_seconds_total", lambda v: "n/a" if v is None else f"{v:.0f}"),
            ("Extra evidence seconds per draft (total spread over all drafts)", "evidence_seconds_per_draft", lambda v: _fmt(v, 1)),
            ("Drafts with unsupported numbers", "unsupported_numbers_rate", _pct),
            ("Drafts flagged for copying a passage", "copy_flag_rate", _pct),
            ("Proposed claims that survive verification", "claim_acceptance", _pct),
            ("Drafts that needed a second attempt", "second_attempt_rate", _pct),
            ("Retrieved passages that fit the requested medium", "passage_fits_medium", _pct),
            ("Mean passage distance (lower is closer)", "mean_passage_distance", lambda v: _fmt(v, 3)),
        ]
        for title, key, fmt in rows:
            cells = [fmt(o.get(key))] + [fmt(effect["by_medium"][m].get(key)) for m in MEDIUMS if m in effect["by_medium"]]
            lines.append(f"| {title} | " + " | ".join(cells) + " |")
        lines += [
            "",
            "How to read the last two rows: by design (Kevin's v1 retriever) retrieval searches by topic only and does not filter "
            "by the requested medium, so every medium of a topic gets the same passages. A low fit means a LinkedIn or X draft is "
            "grounded mostly in blog text. The index holds only 40 transcript chunks, so a talk draft almost never gets one. "
            "A low fit is a description of that design, not by itself a defect.",
        ]
    return "\n".join(lines) + "\n"


def _cmd_slim(args) -> None:
    n = 0
    with open(args.raw, encoding="utf-8-sig") as src, open(args.out, "w", encoding="utf-8") as dst:
        for line in src:
            if line.strip():
                dst.write(json.dumps(slim_draft(json.loads(line)), ensure_ascii=False) + "\n")
                n += 1
    print(f"Wrote {n} slim drafts to {args.out}")


def _cmd_report(args) -> None:
    a, b = load_drafts(args.a_drafts), load_drafts(args.b_drafts)
    cost = paired_cost(a, b, n_boot=args.n_boot)
    effect = rag_only(b)
    markdown = to_markdown(cost, effect, args.label_a, args.label_b)
    if args.out_md:
        Path(args.out_md).write_text(markdown, encoding="utf-8")
    if args.out_json:
        Path(args.out_json).write_text(json.dumps({"paired": cost, "rag_only": effect}, indent=2), encoding="utf-8")
    print(markdown)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("report", help="Paired cost metrics and retrieval-only metrics.")
    r.add_argument("--a-drafts", required=True, help="Drafts of the run without retrieval (raw or slim JSONL).")
    r.add_argument("--b-drafts", required=True, help="Drafts of the run with retrieval.")
    r.add_argument("--label-a", default="RAG off")
    r.add_argument("--label-b", default="RAG on")
    r.add_argument("--n-boot", type=int, default=5000)
    r.add_argument("--out-md")
    r.add_argument("--out-json")
    r.set_defaults(func=_cmd_report)
    s = sub.add_parser("slim", help="Keep only the fields the report reads.")
    s.add_argument("--raw", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(func=_cmd_slim)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
