"""Save a LoRA checkpoint whenever an eval metric improves.

Tracked metrics are overall ``eval_loss``, per-medium loss (blog, LinkedIn,
X, talk), and ``eval_loss_average`` (the unweighted mean of the medium
losses). Lower is better. Each improvement is written under
``best_metrics/<metric>/step-<n>`` and is not rotated away by the trainer's
``save_total_limit``.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

try:
    from transformers import TrainerCallback
except ImportError:  # lightweight test env has no transformers
    class TrainerCallback:  # type: ignore[no-redef]
        pass

MEDIUMS = ("blog", "linkedin", "x", "talk")


def medium_name(source: str) -> str:
    """Collapse scraped source ids onto the four training mediums."""
    name = (source or "").strip()
    if name in ("youtube_jason", "youtube_saastr", "talk"):
        return "talk"
    if name in MEDIUMS:
        return name
    return "other"


def improved_metrics(best: dict[str, float], scores: dict[str, float]) -> list[str]:
    """Record new lows and return the metric names that improved."""
    won: list[str] = []
    for name, value in scores.items():
        previous = best.get(name)
        if previous is None or value < previous:
            best[name] = value
            won.append(name)
    return won


def macro_average(scores: dict[str, float]) -> float | None:
    present = [scores[f"eval_loss_{name}"] for name in MEDIUMS if f"eval_loss_{name}" in scores]
    if not present:
        return None
    return sum(present) / len(present)


class BestMetricCheckpointCallback(TrainerCallback):
    """Copy the adapter each time a tracked eval metric sets a new best."""

    def __init__(self, output_dir: Path, tokenizer, eval_dataset, mediums: list[str]) -> None:
        self.root = Path(output_dir)
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.mediums = mediums
        self.best: dict[str, float] = {}
        self.paths: dict[str, str] = {}
        self.trainer = None
        self._busy = False

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or "loss" not in logs:
            return
        lr = logs.get("learning_rate")
        lr_bit = f" lr={lr:.2e}" if isinstance(lr, float) else ""
        print(
            f"[finetune] STATUS step={state.global_step} "
            f"epoch={state.epoch:.3f} loss={logs['loss']:.4f}{lr_bit}",
            flush=True,
        )
        if state.global_step % 50 == 0:
            _warn_if_spilling()

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if self._busy or self.trainer is None:
            return
        metrics = metrics or {}
        if "eval_loss" not in metrics:
            return
        self._busy = True
        try:
            scores: dict[str, float] = {"eval_loss": float(metrics["eval_loss"])}
            try:
                scores.update(self._medium_losses())
            except Exception as exc:
                print(
                    f"[finetune] METRICS medium split failed ({exc!r}); keeping eval_loss only",
                    flush=True,
                )
            average = macro_average(scores)
            if average is not None:
                scores["eval_loss_average"] = average
            parts = " ".join(f"{name}={value:.4f}" for name, value in scores.items())
            print(f"[finetune] METRICS step={state.global_step} {parts}", flush=True)
            model = kwargs.get("model") or self.trainer.model
            for name in improved_metrics(self.best, scores):
                path = self._save(model, name, state.global_step, scores[name])
                print(
                    f"[finetune] NEW BEST {name}={scores[name]:.4f} "
                    f"step={state.global_step} saved={path}",
                    flush=True,
                )
        finally:
            self._busy = False
            self.trainer.model.train()

    def _tokenized_eval_dataset(self):
        """The trainer's eval rows, already converted to input_ids.

        The original dataset still holds raw ``text``. Passing that to the
        trainer collator raises because the collator pads ``input_ids``.
        """
        prepared = getattr(self.trainer, "eval_dataset", None)
        columns = getattr(prepared, "column_names", None) or []
        if "input_ids" not in columns or len(prepared) != len(self.mediums):
            print(
                "[finetune] METRICS medium split skipped: tokenized eval rows "
                "do not line up with the medium labels",
                flush=True,
            )
            return None
        return prepared

    def _medium_losses(self) -> dict[str, float]:
        if not self.mediums:
            return {}
        dataset = self._tokenized_eval_dataset()
        if dataset is None:
            return {}
        losses: dict[str, float] = {}
        model = self.trainer.model
        model.eval()
        for medium in MEDIUMS:
            indices = [i for i, name in enumerate(self.mediums) if name == medium]
            if not indices:
                continue
            subset = dataset.select(indices)
            total = 0.0
            seen = 0
            for batch in self.trainer.get_eval_dataloader(subset):
                batch = self.trainer._prepare_inputs(batch)
                loss = self.trainer.compute_loss(model, batch)
                size = batch["input_ids"].shape[0]
                total += float(loss.detach()) * size
                seen += size
            if seen:
                losses[f"eval_loss_{medium}"] = total / seen
        return losses

    def _save(self, model, metric: str, step: int, value: float) -> Path:
        dest = self.root / metric / f"step-{step}"
        dest.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(dest))
        self.tokenizer.save_pretrained(str(dest))
        (dest / "best_metric.json").write_text(
            json.dumps({"metric": metric, "step": step, "value": value}, indent=2) + "\n",
            encoding="utf-8",
        )
        self.paths[metric] = str(dest)
        (self.root / "index.json").write_text(
            json.dumps({"best": self.best, "paths": self.paths}, indent=2) + "\n",
            encoding="utf-8",
        )
        return dest


def _warn_if_spilling() -> None:
    try:
        import torch

        if not torch.cuda.is_available():
            return
        free_b, total_b = torch.cuda.mem_get_info()
        reserved_b = torch.cuda.memory_reserved()
        print(
            f"[finetune] VRAM free={free_b / 1e9:.2f}GB "
            f"reserved={reserved_b / 1e9:.2f}GB total={total_b / 1e9:.2f}GB",
            flush=True,
        )
        if reserved_b > total_b:
            print(
                "[finetune] VRAM SPILL reserved memory exceeds the card. "
                "Training is using system RAM.",
                flush=True,
            )
    except Exception:
        return
