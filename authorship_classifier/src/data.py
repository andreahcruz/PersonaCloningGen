"""Loaders. Everything reads local copies in inputs/ and never writes outside outputs/."""
from __future__ import annotations

import json
import pathlib
from collections import defaultdict

import pandas as pd

from textprep import clean_pg_essay, first_window, windows

ROOT = pathlib.Path(__file__).resolve().parents[1]
INP = ROOT / "inputs"


def _jl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


# ---------------------------------------------------------------- Lemkin side
def load_lemkin_rows():
    rows = _jl(INP / "split" / "dataset.jsonl")
    asg = [a for a in _jl(INP / "split" / "split_assignments.jsonl") if a.get("record_type") != "meta"]
    assert len(rows) == len(asg), "dataset and split_assignments must align row by row"
    disp = _jl(INP / "split" / "v3_dispositions.jsonl")
    return rows, asg, blocked_groups(asg, disp)


def blocked_groups(asg, disp):
    """Same rule as Kevin's train_only_index.blocked_group_ids: families of the 30 gold topics."""
    keys, groups = set(), set()
    for d in disp:
        if "gold_overlap_family" in (d.get("reasons") or []):
            keys.add((d["source_file"], int(d["source_line"])))
            groups.update(d.get("prior_group_ids") or [])
    for a in asg:
        if (a.get("source_file"), int(a["source_line"])) in keys:
            groups.add(a["group_id"])
    return groups


def lemkin_windows(medium: str, split: str, size: int, *, first_only: bool, roles=None, max_per_row=None):
    """Real Lemkin text as fixed-length windows.

    first_only=True  -> first window of each post opening (what a draft's first N words are)
    first_only=False -> every window (used for training)
    Each window carries a 'doc' id (source file and line) so uncertainty can be
    resampled by document, not by window.
    """
    rows, asg, blocked = load_lemkin_rows()
    out = []
    for row, a in zip(rows, asg):
        if a["source_platform"] != medium or a["split"] != split or a["group_id"] in blocked:
            continue
        if roles and a.get("sft_role") not in roles:
            continue
        doc = f'{a["source_file"]}:{a["source_line"]}'
        text = row.get("output") or ""
        if first_only:
            w = first_window(text, size)
            ws = [w] if w else []
        else:
            ws = windows(text, size, max_per_row)
        out += [{"text": w, "doc": doc, "label": 1, "src": "lemkin"} for w in ws]
    return out


# ---------------------------------------------------------------- other writer
def pg_windows(split: str, size: int):
    """Paul Graham essays, split by essay using the repo's own splits.json (169, 21, 22)."""
    ids = {str(i) for i in json.load(open(INP / "pg_splits.json"))[{"train": "train", "validation": "val", "test": "test"}[split]]}
    df = pd.read_csv(INP / "cleaned_pg.csv")
    out = []
    for _, r in df.iterrows():
        if str(r["id"]) in ids:
            out += [{"text": w, "doc": f'pg:{r["id"]}', "label": 0, "src": "pg"}
                    for w in windows(clean_pg_essay(str(r["text"])), size)]
    return out


# ---------------------------------------------------------------- model drafts
SYSTEMS = [
    # (label, system, seed, file, text key, id key)
    ("A_base_llama", "A Base Llama", 42, "A_base_llama.jsonl"),
    ("C_personarag_seed42", "C PersonaRAG stand-in", 42, "C_personarag_seed42.jsonl"),
    ("D_repaired_seed42", "D QLoRA repaired", 42, "D_repaired_seed42.jsonl"),
    ("D_repaired_seed43", "D QLoRA repaired", 43, "D_repaired_seed43.jsonl"),
    ("D_repaired_seed44", "D QLoRA repaired", 44, "D_repaired_seed44.jsonl"),
    ("E_alone_seed42", "E Relabel, no retrieval", 42, "E_alone_seed42.jsonl"),
    ("E_alone_seed43", "E Relabel, no retrieval", 43, "E_alone_seed43.jsonl"),
    ("E_alone_seed44", "E Relabel, no retrieval", 44, "E_alone_seed44.jsonl"),
    ("E_rag_seed42", "E Relabel + retrieval", 42, "E_rag_seed42.jsonl"),
    ("E_rag_seed43", "E Relabel + retrieval", 43, "E_rag_seed43.jsonl"),
    ("E_rag_seed44", "E Relabel + retrieval", 44, "E_rag_seed44.jsonl"),
    ("F_REL", "F Model F (F-REL)", 42, "F_REL.jsonl"),
]


def load_drafts():
    """Returns a flat list of dicts: run, system, seed, topic, medium, text."""
    out = []
    for run, system, seed, fname in SYSTEMS:
        path = INP / "kevin_drafts" / fname
        if not path.exists():              # some inputs are not in the repository (Colab, private Model F drafts)
            continue
        for r in _jl(path):
            text = r.get("answer") if "answer" in r else r.get("response")
            topic = r.get("id") or r.get("question_id")
            if text is None or topic is None:
                raise KeyError(f"{fname}: cannot find draft text or topic id in {sorted(r)[:12]}")
            out.append({"run": run, "system": system, "seed": seed, "topic": topic,
                        "medium": r["medium"], "text": text})
    return out


if __name__ == "__main__":
    for medium, size in (("blog", 100), ("linkedin", 60)):
        for split in ("train", "validation", "test"):
            tr = lemkin_windows(medium, split, size, first_only=(split != "train"))
            pg = pg_windows(split, size)
            print(f"{medium:9s} W={size:3d} {split:10s} lemkin windows {len(tr):6d} "
                  f"({len({w['doc'] for w in tr}):5d} docs) | PG windows {len(pg):5d} ({len({w['doc'] for w in pg}):3d} essays)")
    d = load_drafts()
    import collections
    c = collections.Counter((x["run"], x["medium"]) for x in d)
    print("drafts per run:", {k: v for k, v in sorted(c.items()) if k[1] == "blog"})
