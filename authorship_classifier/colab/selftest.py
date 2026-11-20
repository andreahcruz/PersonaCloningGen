"""Runs the notebook's own code cells against a tiny random Llama model on the CPU.
It checks the logic (prompts, batching, stop handling, checkpoint and resume, output format),
not the quality of the text. Run:  python -X utf8 colab/selftest.py"""
import json, pathlib, shutil, sys, tempfile, types

HERE = pathlib.Path(__file__).resolve().parent
nb = json.load(open(HERE / "make_llm_negatives.ipynb", encoding="utf-8"))
code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
# cell order: 0 pip, 1 config, 2 topics, 3 load model, 4 functions, 5 run, 6 check, 7 save, 8 drive

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ns = {}
exec(code[1], ns)                                  # config
tmp = pathlib.Path(tempfile.mkdtemp())
ns["OUT_DIR"] = str(tmp)
ns["JSONL"] = str(tmp / "llm_negatives.jsonl")
exec(code[2], ns)                                  # topics
assert len(ns["TOPICS"]) == 450 and len(set(ns["TOPICS"])) == 450

name = "hf-internal-testing/tiny-random-LlamaForCausalLM"
tok = AutoTokenizer.from_pretrained(name)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.chat_template = ("{{ bos_token }}{% for m in messages %}<|start_header_id|>{{ m['role'] }}<|end_header_id|>\n\n"
                     "{{ m['content'] }}<|eot_id|>{% endfor %}{% if add_generation_prompt %}"
                     "<|start_header_id|>assistant<|end_header_id|>\n\n{% endif %}")
model = AutoModelForCausalLM.from_pretrained(name).eval()
ns.update(tok=tok, model=model, mode="selftest", BATCH=4, torch=torch)
ns["MAX_NEW"] = {"blog": 12, "linkedin": 8, "talk": 12}
ns["PLAN"] = {"blog": (0, 6), "talk": (0, 3), "linkedin": (3, 7)}

# the functions cell (index 4) and the config cell must run against these stand-ins
exec(code[4], ns)
for medium in ("blog", "linkedin", "talk"):
    ns["run_medium"](medium)

rows = [json.loads(l) for l in open(ns["JSONL"], encoding="utf-8")]
assert len(rows) == 6 + 3 + 4, len(rows)
need = {"id", "medium", "topic", "prompt", "seed", "model", "precision", "text", "new_tokens", "stop"}
assert all(need <= set(r) for r in rows)
assert all(r["prompt"].startswith("Write a") and r["topic"] in r["prompt"] for r in rows)
assert all(r["stop"] in ("eos", "length") for r in rows)
assert {r["medium"] for r in rows} == {"blog", "linkedin", "talk"}
assert len({r["id"] for r in rows}) == len(rows), "duplicate ids"
print("run ok:", len(rows), "rows; stops:", {s: sum(r["stop"] == s for r in rows) for s in ("eos", "length")})

# resume: a second run must add nothing
before = len(rows)
for medium in ("blog", "linkedin", "talk"):
    ns["run_medium"](medium)
assert len([1 for _ in open(ns["JSONL"], encoding="utf-8")]) == before
print("resume ok: second run added nothing")

# resume after a partial file: drop the last 3 rows, run again, the same ids come back
lines = open(ns["JSONL"], encoding="utf-8").read().splitlines()
open(ns["JSONL"], "w", encoding="utf-8").write("\n".join(lines[:-3]) + "\n")
for medium in ("blog", "linkedin", "talk"):
    ns["run_medium"](medium)
assert len([1 for _ in open(ns["JSONL"], encoding="utf-8")]) == before
print("partial-resume ok")

# stop handling: a draft that ends with the end-of-text marker must be trimmed there and labelled "eos"
class FakeModel:
    device = "cpu"
    def generate(self, input_ids, attention_mask, **kw):
        eos = ns["EOS_IDS"][0]
        n = input_ids.shape[0]
        tail = torch.tensor([[11, 12, eos, 13, 14]] * n)       # text, end marker, then junk after it
        tail[0] = torch.tensor([11, 12, 13, 14, 15])            # row 0 never ends: hits the limit
        return torch.cat([input_ids, tail], dim=1)
ns["model"] = FakeModel()
res = ns["generate_batch"]("blog", ns["TOPICS"][:2], 1)
assert res[0]["stop"] == "length" and res[0]["new_tokens"] == 5, res[0]
assert res[1]["stop"] == "eos" and res[1]["new_tokens"] == 2, res[1]
print("eos trimming ok:", {k: res[1][k] for k in ("stop", "new_tokens")})

# the file feeds the classifier loader format: text + medium fields
sys.path.insert(0, str(HERE.parent))
import run_pipeline as rp
rp.EXTERNAL_NEG = [json.loads(l) for l in open(ns["JSONL"], encoding="utf-8")]
assert rp.llm_negatives([], "blog", 3, set()) is not None
print("pipeline can read it: ok")
shutil.rmtree(tmp)
print("SELFTEST PASSED")
