"""Collect the inputs for the authorship classifier from the repository. Read only.

Run from anywhere inside a clone that has fetched all branches:
    git fetch --all
    python authorship_classifier/prepare_inputs.py

Public inputs (copied with git show, nothing is modified):
  origin/kevin/lightrag-model-f : split files, gold eval, saved drafts for Models A, D and E
  origin/master                 : Paul Graham essays and the repo's own essay split
  this checkout                 : Kevin's fixed-length voice distances from PUK-28

NOT in the repository (the pipeline skips them if absent):
  PersonaRAG and Model E with retrieval seeds 43 and 44 : Colab outputs on the team Drive
        --colab-dir  folder that contains EXP-COLAB-* directories (puk28_results)
  Model F drafts : Kevin's private export (the repo is public, so never commit it)
        --f-rel      path to generations.jsonl
"""
from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess

HERE = pathlib.Path(__file__).resolve().parent
REPO = pathlib.Path(subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=HERE, capture_output=True, text=True, check=True).stdout.strip())
INP = HERE / "inputs"
KEVIN = "origin/kevin/lightrag-model-f"
EXP = "experiments/"


def show(ref: str, path: str, dest: pathlib.Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    out = subprocess.run(["git", "show", f"{ref}:{path}"], cwd=REPO, capture_output=True)
    if out.returncode != 0:
        raise SystemExit(f"cannot read {ref}:{path}. Did you run 'git fetch --all'?")
    dest.write_bytes(out.stdout)
    print("copied", dest.relative_to(HERE))


def find_dir(prefix: str) -> str:
    names = subprocess.run(["git", "ls-tree", "--name-only", KEVIN, EXP], cwd=REPO, capture_output=True, text=True).stdout.split()
    return next(n for n in names if n.startswith(EXP + prefix))


def main():
    ap = argparse.ArgumentParser(description="Collect inputs for the authorship classifier (read only).")
    ap.add_argument("--colab-dir", type=pathlib.Path)
    ap.add_argument("--f-rel", type=pathlib.Path)
    a = ap.parse_args()

    r004 = "EXP-20261003-004-relabel-continuations/raw/"
    for name in ("dataset.jsonl", "split_assignments.jsonl"):
        show(KEVIN, EXP + r004 + name, INP / "split" / name)
    show(KEVIN, EXP + "EXP-20261003-003-complete-source-sft-v3/raw/dispositions.jsonl", INP / "split" / "v3_dispositions.jsonl")
    show(KEVIN, "data/openai_ft/lemkin_gold_eval.jsonl", INP / "gold_eval.jsonl")
    show("origin/master", "cleaned_pg.csv", INP / "cleaned_pg.csv")
    show("origin/master", "splits.json", INP / "pg_splits.json")

    d = INP / "kevin_drafts"
    show(KEVIN, EXP + "EXP-20261001-010-repaired-and-baselines/raw/llama3.1.jsonl", d / "A_base_llama.jsonl")
    for prefix, seed in (("EXP-20261004-003-", 42), ("EXP-20261004-005-", 43), ("EXP-20261004-006-", 44)):
        base = find_dir(prefix)
        show(KEVIN, f"{base}/raw/relabel.jsonl", d / f"E_alone_seed{seed}.jsonl")
        show(KEVIN, f"{base}/raw/repaired.jsonl", d / f"D_repaired_seed{seed}.jsonl")
    show(KEVIN, EXP + "EXP-20261005-003-relabel-rag-final/raw/relabel.jsonl", d / "E_rag_seed42.jsonl")

    vf = REPO / "docs" / "evaluation" / "examples" / "data" / "voice_fixed"
    (INP / "puk28_voice_fixed").mkdir(parents=True, exist_ok=True)
    for f in vf.glob("*.voice_fixed.jsonl"):
        shutil.copy(f, INP / "puk28_voice_fixed" / f.name)
        print("copied", f.name)

    if a.colab_dir:
        for src, dst in (("EXP-COLAB-relabel-rag-seed43/raw/relabel.jsonl", "E_rag_seed43.jsonl"),
                         ("EXP-COLAB-relabel-rag-seed44/raw/relabel.jsonl", "E_rag_seed44.jsonl"),
                         ("EXP-COLAB-personarag-newdata-seed42/raw/llama3.1.jsonl", "C_personarag_seed42.jsonl")):
            shutil.copy(a.colab_dir / src, d / dst)
            print("copied", dst)
    else:
        print("note: no --colab-dir, so PersonaRAG (C) and E+retrieval seeds 43 and 44 are skipped")
    if a.f_rel:
        shutil.copy(a.f_rel, d / "F_REL.jsonl")
        print("copied F_REL.jsonl (PRIVATE, never commit)")
    else:
        print("note: no --f-rel, so Model F is skipped")


if __name__ == "__main__":
    main()
