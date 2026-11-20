"""Checks the file that comes back from the Colab notebook BEFORE it is used.
Run:  python -X utf8 validate_negatives.py llm_negatives_bundle.zip      (or a .jsonl)
Writes outputs_colab/negatives_validation.json and outputs_colab/llm_negatives_clean.jsonl.

Pass/fail rules are fixed here, before any result is seen:
  - the checksum in manifest.json matches the file
  - no topic overlaps a gold topic (must be 0)
  - at least 300 blog, 200 LinkedIn and 100 talk texts survive cleaning
  - refusals at most 2 percent, texts sharing a 12-word run with Lemkin training posts at most 5 percent
    (12 words is Kevin's own copy gate. Calibrated on the existing Base Llama drafts: 2 of 90 share a 12-word run,
    but 3 to 17 percent share an 8-word run, which is just common phrases, so 8 words is too strict).
    Spans that also occur in the prompt or topic title are ignored, exactly as in Kevin's copy gate. The first version
    of this rule did not do that and flagged 48 texts that only repeated the topic title (a real Lemkin post title);
    the rule was corrected after that first run, and the raw number is still reported as copy_share_raw.
  - at least 90 percent of generated texts are usable (long enough, not empty, not a refusal)
"""
import collections, hashlib, io, json, pathlib, re, statistics, sys, zipfile

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from data import INP, load_lemkin_rows
from textprep import ngram_set, normalize

MIN_WORDS = {"blog": 150, "talk": 150, "linkedin": 60}
MIN_KEPT = {"blog": 300, "linkedin": 200, "talk": 100}
REFUSAL = re.compile(r"\bI (can't|cannot|can not|won't)\b|I'm not able to|as an AI", re.I)
norm = lambda s: re.sub(r"[^a-z0-9 ]", "", s.lower().replace("’", "'")).strip()


def jac(a, b):
    A, B = set(norm(a).split()), set(norm(b).split())
    return len(A & B) / max(1, len(A | B))


def load(path):
    path = pathlib.Path(path)
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            raw = z.read("llm_negatives.jsonl")
            manifest = json.loads(z.read("manifest.json")) if "manifest.json" in z.namelist() else None
    else:
        raw, manifest = path.read_bytes(), None
    rows = [json.loads(l) for l in raw.decode("utf-8").splitlines() if l.strip()]
    return rows, manifest, hashlib.sha256(raw).hexdigest()


def main(path, out_dir="outputs_colab"):
    rows, manifest, sha = load(path)
    out = ROOT / out_dir
    out.mkdir(exist_ok=True)
    gold = [json.loads(l)["topic"] for l in open(INP / "gold_eval.jsonl", encoding="utf-8")]
    ds, asg, _ = load_lemkin_rows()
    train_grams = set()
    for row, a in zip(ds, asg):
        if a["split"] == "train":
            train_grams |= ngram_set(row.get("output") or "", 12)

    checks, clean = {}, []
    checks["checksum_matches_manifest"] = None if not manifest else (manifest["jsonl_sha256"] == sha)
    gold_hits = sorted({r["topic"] for r in rows if max(jac(r["topic"], g) for g in gold) >= 0.5 or norm(r["topic"]) in {norm(g) for g in gold}})
    checks["gold_topic_overlaps"] = len(gold_hits)
    seen = collections.Counter(norm(r["text"])[:300] for r in rows)
    flags = collections.Counter()
    for r in rows:
        w = len(r["text"].split())
        bad = None
        if w == 0: bad = "empty"
        elif w < MIN_WORDS[r["medium"]]: bad = "too_short"
        elif REFUSAL.search(r["text"][:200]): bad = "refusal"
        elif seen[norm(r["text"])[:300]] > 1: bad = "duplicate"
        elif (ngram_set(r["text"], 12) & train_grams) - (ngram_set(r["prompt"], 12) | ngram_set(r["topic"], 12)): bad = "copies_lemkin_train_12gram"
        flags[bad or "ok"] += 1
        r["_flag"] = bad or "ok"
        if not bad:
            clean.append({k: v for k, v in r.items() if k != "_flag"})
    n = len(rows)
    kept = collections.Counter(r["medium"] for r in clean)
    checks["n_rows"] = n
    checks["flags"] = dict(flags)
    checks["kept_per_medium"] = dict(kept)
    checks["usable_share"] = round(len(clean) / max(1, n), 3)
    checks["refusal_share"] = round(flags["refusal"] / max(1, n), 3)
    checks["copy_share"] = round(flags["copies_lemkin_train_12gram"] / max(1, n), 3)
    checks["copy_share_raw_incl_topic_title_repeats"] = round(sum(bool(ngram_set(r["text"], 12) & train_grams) for r in rows) / max(1, n), 3)
    checks["hit_token_limit_share"] = round(sum(r["stop"] == "length" for r in rows) / max(1, n), 3)
    words = collections.defaultdict(list)
    for r in rows: words[r["medium"]].append(len(r["text"].split()))
    checks["median_words"] = {m: statistics.median(v) for m, v in words.items()}
    verdict = {
        "checksum": checks["checksum_matches_manifest"] in (True, None),
        "no_gold_overlap": checks["gold_topic_overlaps"] == 0,
        "enough_per_medium": all(kept[m] >= MIN_KEPT[m] for m in MIN_KEPT),
        "refusals_le_2pct": checks["refusal_share"] <= 0.02,
        "copying_le_5pct": checks["copy_share"] <= 0.05,
        "usable_ge_90pct": checks["usable_share"] >= 0.90,
    }
    checks["verdict"] = verdict
    checks["PASS"] = all(verdict.values())
    json.dump(checks, open(out / "negatives_validation.json", "w", encoding="utf-8"), indent=2)
    with open(out / "llm_negatives_clean.jsonl", "w", encoding="utf-8") as f:
        for r in clean: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(checks, indent=2))
    print("\nRESULT:", "PASS" if checks["PASS"] else "FAIL", "| clean file:", out / "llm_negatives_clean.jsonl")
    return checks


if __name__ == "__main__":
    main(sys.argv[1], *(sys.argv[2:3]))
