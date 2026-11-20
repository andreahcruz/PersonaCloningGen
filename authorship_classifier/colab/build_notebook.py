"""Builds make_llm_negatives.ipynb. Run:  python -X utf8 colab/build_notebook.py
The notebook generates plain Llama 3.1 8B text on NON-gold topics. The classifier uses it as the
'plain LLM text' negative class (python run_pipeline.py --llm-negatives llm_negatives.jsonl)."""
import json, pathlib

HERE = pathlib.Path(__file__).resolve().parent
TOPICS = json.load(open(HERE / "topics.json", encoding="utf-8"))
cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n").splitlines(keepends=True)})


md("""
# Plain Llama text for the authorship classifier (DATA 298B, PUK-28)

**What this does.** Writes about 900 short pieces with plain Llama 3.1 8B Instruct (no fine-tuning, no retrieval) on
450 Lemkin blog titles that are NOT among our 30 gold topics. The classifier uses this text as its "plain LLM writing"
negatives, so none of our systems' drafts are needed for training. It uses the same prompt wording and settings as our
Base Llama baseline (temperature 0.7, top-p 0.9, repetition penalty 1.15, limits 768 / 320 / 768 tokens).

**How to run.** Runtime > Change runtime type > GPU (A100 if you have it, otherwise it falls back to 4-bit on a smaller
GPU, which is slower). Then Runtime > Run all. Nothing to upload and no Hugging Face login needed.

**How long.** About 30 to 60 minutes on an A100 (about 10 minutes of that is the model download). Much longer on a T4.
If Colab disconnects, run all again: finished items are skipped.

**What to send back.** The last cell downloads `llm_negatives_bundle.zip`. Put it in the `authorship_classifier`
folder on the laptop and tell Claude. Nothing private is in it: it is Llama writing, not Lemkin's posts.
""")

code("""
!pip -q install -U "transformers>=4.44" accelerate bitsandbytes
""")

code('''
import collections, hashlib, json, os, platform, re, statistics, time, zipfile

MODEL_ID = "unsloth/Meta-Llama-3.1-8B-Instruct"   # open copy of Meta Llama 3.1 8B Instruct, no login needed
SEED = 42
OUT_DIR = "/content/llm_negatives"
os.makedirs(OUT_DIR, exist_ok=True)
JSONL = f"{OUT_DIR}/llm_negatives.jsonl"

# Same wording as the Base Llama baseline (experiment EXP-20261001-010)
INSTRUCTION = {
    "blog": "Write a blog post in the style of Jason Lemkin about: {topic}",
    "linkedin": "Write a LinkedIn post in the style of Jason Lemkin about: {topic}",
    "talk": "Write a talk in the style of Jason Lemkin about: {topic}",
}
MAX_NEW = {"blog": 768, "linkedin": 320, "talk": 768}
SAMPLING = dict(do_sample=True, temperature=0.7, top_p=0.9, repetition_penalty=1.15)
# which slice of TOPICS each medium uses (blog and talk share the first 150 topics, LinkedIn uses the rest)
PLAN = {"blog": (0, 450), "talk": (0, 150), "linkedin": (150, 450)}
''')

md("""
### Topics
450 titles of real Lemkin blog posts from the training split. Removed: the 30 gold topics, their near copies and families,
event promos, and duplicates. The list is built into this notebook, so there is nothing to upload.
""")

code("TOPICS = json.loads(r'''" + json.dumps(TOPICS, ensure_ascii=False) + "''')\n"
     "print(len(TOPICS), 'topics. First three:')\n"
     "for t in TOPICS[:3]:\n"
     "    print(' -', t)\n")

md("### Load the model")
code('''
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

assert torch.cuda.is_available(), "No GPU. Use Runtime > Change runtime type > GPU, then Run all again."
gpu_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
print(torch.cuda.get_device_name(0), f"{gpu_gb:.0f} GB")

tok = AutoTokenizer.from_pretrained(MODEL_ID)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
if gpu_gb >= 30:
    mode, BATCH = "bf16", 32
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda").eval()
else:
    mode, BATCH = "4bit", 8
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="cuda", quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True)).eval()
print("loaded in", mode, "| batch size", BATCH)
''')

md("### Generation functions")
code('''
EOS_IDS = [i for i in {tok.eos_token_id, tok.convert_tokens_to_ids("<|eot_id|>")} if i is not None]


def build_prompt(medium, topic):
    message = INSTRUCTION[medium].format(topic=topic)
    text = tok.apply_chat_template([{"role": "user", "content": message}], tokenize=False, add_generation_prompt=True)
    return text, message


@torch.inference_mode()
def generate_batch(medium, topics, seed):
    torch.manual_seed(seed)
    prompts = [build_prompt(medium, t)[0] for t in topics]
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    out = model.generate(**enc, max_new_tokens=MAX_NEW[medium], eos_token_id=EOS_IDS,
                         pad_token_id=tok.pad_token_id, **SAMPLING)
    results = []
    for row in out[:, enc["input_ids"].shape[1]:]:
        ids, stop = row.tolist(), "length"
        for i, x in enumerate(ids):
            if x in EOS_IDS:
                ids, stop = ids[:i], "eos"
                break
        ids = [x for x in ids if x != tok.pad_token_id]
        results.append({"text": tok.decode(ids, skip_special_tokens=True).strip(), "new_tokens": len(ids), "stop": stop})
    return results


def finished_ids():
    done = set()
    if os.path.exists(JSONL):
        with open(JSONL, encoding="utf-8") as f:
            done = {json.loads(line)["id"] for line in f if line.strip()}
    return done


def run_medium(medium):
    lo, hi = PLAN[medium]
    items = [(f"{medium}-{i:04d}", TOPICS[i], i) for i in range(lo, hi)]
    done = finished_ids()
    todo = [x for x in items if x[0] not in done]
    print(f"{medium}: {len(items)} planned, {len(done & {x[0] for x in items})} already done, {len(todo)} to do")
    t0 = time.time()
    for b in range(0, len(todo), BATCH):
        chunk = todo[b:b + BATCH]
        seed = SEED + 1000 * list(PLAN).index(medium) + chunk[0][2]     # same seed whether or not we resumed
        res = generate_batch(medium, [c[1] for c in chunk], seed)
        with open(JSONL, "a", encoding="utf-8") as f:
            for (item_id, topic, _), r in zip(chunk, res):
                f.write(json.dumps({"id": item_id, "medium": medium, "topic": topic,
                                    "prompt": INSTRUCTION[medium].format(topic=topic), "seed": seed,
                                    "model": MODEL_ID, "precision": mode, **r}, ensure_ascii=False) + "\\n")
        print(f"  {medium} {min(b + BATCH, len(todo))}/{len(todo)}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
''')

md("### Run (this is the long step)")
code('''
for medium in ("blog", "linkedin", "talk"):
    run_medium(medium)
print("done")
''')

md("### Quick check of what came out")
code('''
rows = [json.loads(line) for line in open(JSONL, encoding="utf-8") if line.strip()]
by = collections.defaultdict(list)
for r in rows:
    by[r["medium"]].append(r)
refusal = re.compile(r"\\bI (can't|cannot|can not|won't)\\b|I'm not able to|as an AI", re.I)
for m, v in by.items():
    words = [len(r["text"].split()) for r in v]
    print(f"{m:9s} n={len(v):3d} median words={statistics.median(words):5.0f} min={min(words):4d} "
          f"hit token limit={sum(r['stop'] == 'length' for r in v):3d} empty={sum(w == 0 for w in words)} "
          f"possible refusals={sum(bool(refusal.search(r['text'][:200])) for r in v)}")
print()
for m, v in by.items():
    print("SAMPLE", m, "->", v[0]["text"][:350].replace("\\n", " "), "\\n")
''')

md("### Save and download (nothing private inside: it is Llama writing)")
code('''
sha = hashlib.sha256(open(JSONL, "rb").read()).hexdigest()
manifest = {
    "purpose": "plain-LLM negatives for the DATA 298B authorship classifier (PUK-28)",
    "model": MODEL_ID, "precision": mode, "seed": SEED, "sampling": SAMPLING, "max_new_tokens": MAX_NEW,
    "instruction_templates": INSTRUCTION, "plan": PLAN, "n_topics": len(TOPICS),
    "n_rows": len(rows), "rows_per_medium": {m: len(v) for m, v in by.items()},
    "jsonl_sha256": sha, "gpu": torch.cuda.get_device_name(0),
    "torch": torch.__version__, "python": platform.python_version(),
    "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "note": "Same prompt wording and sampling as the Base Llama baseline EXP-20261001-010, but the HF fp16/4-bit weights, "
            "not the Ollama Q4_K_M build used for that baseline, so wording will differ slightly.",
}
json.dump(manifest, open(f"{OUT_DIR}/manifest.json", "w"), indent=2)
bundle = "/content/llm_negatives_bundle.zip"
with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
    z.write(JSONL, "llm_negatives.jsonl")
    z.write(f"{OUT_DIR}/manifest.json", "manifest.json")
print("rows:", len(rows), "| sha256:", sha[:16], "...")
from google.colab import files
files.download(bundle)
''')

md("""
### Optional: also keep a copy on Google Drive
Off by default, so "Run all" never asks you to sign in. Set `SAVE_TO_DRIVE = True` and run this cell by itself if you want it.
""")
code('''
SAVE_TO_DRIVE = False
if SAVE_TO_DRIVE:
    from google.colab import drive
    import shutil
    drive.mount("/content/drive")
    os.makedirs("/content/drive/MyDrive/298B_colab", exist_ok=True)
    shutil.copy(bundle, "/content/drive/MyDrive/298B_colab/llm_negatives_bundle.zip")
    print("copied to Drive: 298B_colab/llm_negatives_bundle.zip")
else:
    print("Drive copy skipped (SAVE_TO_DRIVE is False).")
''')

md("""
### After this
Download `llm_negatives_bundle.zip`, tell Claude, and the classifier is rerun with these negatives
(`python run_pipeline.py --llm-negatives llm_negatives.jsonl --out-dir outputs_colab`).
""")

nb = {"nbformat": 4, "nbformat_minor": 5,
      "metadata": {"accelerator": "GPU", "colab": {"gpuType": "A100", "provenance": []},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "cells": cells}
out = HERE / "make_llm_negatives.ipynb"
json.dump(nb, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("wrote", out, "|", len(cells), "cells |", len(TOPICS), "topics")
