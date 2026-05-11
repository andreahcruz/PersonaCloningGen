"""Dump key metadata from two GGUFs and diff the interesting fields.

Used to compare the broken FT GGUF vs Ollama's known-good base llama3.1 GGUF.
"""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path

for _name in ("stdout", "stderr"):
    s = getattr(sys, _name, None)
    if s is not None and hasattr(s, "reconfigure"):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

from gguf import GGUFReader  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]

FT_GGUF = (
    REPO_ROOT
    / "host_finetune"
    / "output"
    / "lemkin-clone_gguf"
    / "Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf"
)
BASE_GGUF = Path(
    os.path.expandvars(
        r"%USERPROFILE%\.ollama\models\blobs\sha256-667b0c1932bc6ffc593ed1d03f895bf2dc8dc6df21db3042284a6f4416b06a29"
    )
)

INTERESTING_KEYS = (
    "general.architecture",
    "general.name",
    "general.quantization_version",
    "general.file_type",
    "llama.context_length",
    "llama.embedding_length",
    "llama.block_count",
    "llama.attention.head_count",
    "llama.attention.head_count_kv",
    "llama.vocab_size",
    "tokenizer.ggml.model",
    "tokenizer.ggml.pre",
    "tokenizer.ggml.bos_token_id",
    "tokenizer.ggml.eos_token_id",
    "tokenizer.ggml.padding_token_id",
    "tokenizer.chat_template",
    "tokenizer.ggml.add_bos_token",
    "tokenizer.ggml.add_eos_token",
)


def _val(reader: GGUFReader, key: str):
    """Return a scalar/array-like preview of a metadata field, or None."""
    field = reader.get_field(key)
    if field is None:
        return None
    if not field.data:
        return None
    try:
        parts = field.parts
        idx = field.data[0]
        data = parts[idx]
        if hasattr(data, "tobytes"):
            try:
                return data.tobytes().decode("utf-8", errors="replace")
            except Exception:
                return repr(data)
        return data
    except Exception as e:
        return f"<err: {e!r}>"


def _dump(path: Path, label: str) -> dict:
    print(f"\n=== {label} ===")
    print(f"path: {path}")
    print(f"size: {path.stat().st_size / 1e9:.2f} GB")
    reader = GGUFReader(str(path))
    out: dict = {}
    for k in INTERESTING_KEYS:
        v = _val(reader, k)
        out[k] = v
        v_str = str(v)
        if len(v_str) > 240:
            v_str = v_str[:240] + f"...[+{len(str(v)) - 240}c]"
        print(f"  {k}: {v_str}")

    print("  -- vocab probe --")
    try:
        tokens_field = reader.get_field("tokenizer.ggml.tokens")
        if tokens_field and tokens_field.data:
            n_tokens = len(tokens_field.data)
            out["n_tokens"] = n_tokens
            print(f"  tokens.count: {n_tokens}")
            samples = [0, 1, 2, 100, 1000, 10000, 100000, 128000, 128001, 128009, n_tokens - 1]
            for i in samples:
                if i >= n_tokens:
                    continue
                idx = tokens_field.data[i]
                part = tokens_field.parts[idx]
                try:
                    s = part.tobytes().decode("utf-8", errors="replace")
                except Exception:
                    s = repr(part)
                print(f"    [{i}]: {s!r}")
    except Exception as e:
        print(f"  vocab probe error: {e!r}")

    print("  -- tensor info --")
    n_tensors = len(reader.tensors)
    out["n_tensors"] = n_tensors
    print(f"  n_tensors: {n_tensors}")
    tmap = {}
    for t in reader.tensors:
        tmap[t.name] = (tuple(int(x) for x in t.shape), t.tensor_type.name, int(t.n_bytes))
    out["tensor_map"] = tmap

    print("  -- weight stats on key tensors --")
    import numpy as np

    def _stats(name: str) -> None:
        for t in reader.tensors:
            if t.name == name:
                try:
                    arr = np.asarray(t.data)
                    flat = arr.astype("float32", copy=False).ravel()
                    n = flat.size
                    if n == 0:
                        print(f"    {name}: EMPTY"); return
                    # sample a slice for speed
                    samp = flat if n <= 200_000 else flat[:: max(1, n // 200_000)]
                    print(
                        f"    {name} (type={t.tensor_type.name} shape={tuple(int(x) for x in t.shape)}): "
                        f"mean={samp.mean():.4f} std={samp.std():.4f} "
                        f"min={samp.min():.4f} max={samp.max():.4f} "
                        f"nan={int(np.isnan(samp).sum())} inf={int(np.isinf(samp).sum())} "
                        f"zero_frac={float((samp == 0).mean()):.3f}"
                    )
                except Exception as e:
                    print(f"    {name}: stats_err {e!r}")
                return
        print(f"    {name}: MISSING")

    for n in (
        "token_embd.weight",
        "output.weight",
        "output_norm.weight",
        "blk.0.attn_norm.weight",
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.0.attn_v.weight",
        "blk.0.attn_output.weight",
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_down.weight",
        "blk.0.ffn_up.weight",
        "blk.31.attn_q.weight",
        "blk.31.ffn_down.weight",
    ):
        _stats(n)

    return out


def main() -> None:
    ft = _dump(FT_GGUF, "FT GGUF (broken)")
    base = _dump(BASE_GGUF, "BASE Ollama llama3.1 (clean)")

    print("\n=== DIFF (FT vs BASE) ===")
    keys = sorted(set(ft) | set(base))
    for k in keys:
        if k == "tensor_map":
            continue
        a, b = ft.get(k), base.get(k)
        if a != b:
            sa = str(a)
            sb = str(b)
            if len(sa) > 200:
                sa = sa[:200] + "..."
            if len(sb) > 200:
                sb = sb[:200] + "..."
            print(f"  [{k}]")
            print(f"    FT:   {sa}")
            print(f"    BASE: {sb}")

    print("\n=== TENSOR DIFF (FT vs BASE) ===")
    ft_tm = ft.get("tensor_map", {}) or {}
    ba_tm = base.get("tensor_map", {}) or {}
    only_ft = sorted(set(ft_tm) - set(ba_tm))
    only_base = sorted(set(ba_tm) - set(ft_tm))
    common = sorted(set(ft_tm) & set(ba_tm))
    print(f"only-in-FT (n={len(only_ft)}):   {only_ft[:20]}")
    print(f"only-in-BASE (n={len(only_base)}): {only_base[:20]}")
    differ = []
    for name in common:
        if ft_tm[name] != ba_tm[name]:
            differ.append((name, ft_tm[name], ba_tm[name]))
    print(f"shape/type-differ (n={len(differ)}):")
    for name, ft_info, ba_info in differ[:30]:
        print(f"  {name}: FT={ft_info} BASE={ba_info}")


if __name__ == "__main__":
    main()
