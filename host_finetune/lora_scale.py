"""Inference-time LoRA contribution. Weights stay where they were loaded.

PEFT 0.19.1 exposes this on the layer, not on PeftModel. ``LoraLayer.set_scale``
writes ``scaling[adapter] = factor * lora_alpha / r`` and does not touch A or B.
Unsloth replaces ``Linear4bit.forward`` with ``fast_lora_forward``, which reads
``self.scaling[active_adapter]`` on every forward. The next sample therefore
sees the new factor. ``set_scale(adapter, 1.0)`` writes the trained scale back.
"""
from __future__ import annotations

import math
from contextlib import contextmanager


def lora_layers(model):
    found = []
    for module in model.modules():
        scaling = getattr(module, "scaling", None)
        if not callable(getattr(module, "set_scale", None)) or not isinstance(scaling, dict):
            continue
        if not scaling:
            continue
        found.append(module)
    return found


class LoraScale:
    """Absolute contribution relative to the trained alpha/r. 1.0 is Relabel."""

    def __init__(self, model):
        self.model = model
        self.layers = lora_layers(model)
        if not self.layers:
            raise RuntimeError("no LoRA layers expose set_scale")
        if any(getattr(layer, "merged", False) for layer in self.layers):
            raise RuntimeError("refusing to scale a merged adapter")
        self.baseline: list[tuple[object, str, float]] = []
        for layer in self.layers:
            for name, value in layer.scaling.items():
                self.baseline.append((layer, name, float(value)))
                layer.set_scale(name, 1.0)
                restored = float(layer.scaling[name])
                if not math.isclose(restored, float(value), rel_tol=1e-6, abs_tol=1e-6):
                    raise RuntimeError(
                        f"set_scale(1.0) changed the trained scale from {value} to {restored}"
                    )
        self._weights = _weight_token(self.layers[0])

    def apply(self, factor: float) -> None:
        if factor < 0:
            raise ValueError("LoRA contribution cannot be negative")
        for layer, name, _original in self.baseline:
            layer.set_scale(name, factor)
        self._check_weights()

    def restore(self) -> None:
        for layer, name, original in self.baseline:
            layer.set_scale(name, 1.0)
            current = float(layer.scaling[name])
            if not math.isclose(current, original, rel_tol=1e-6, abs_tol=1e-6):
                raise RuntimeError(
                    f"LoRA scale restored to {current}, not the trained value {original}"
                )
        self._check_weights()

    def factors(self) -> list[float]:
        """Current scale divided by the trained scale, one entry per layer."""
        found = []
        for layer, name, original in self.baseline:
            current = float(layer.scaling[name])
            found.append(current / original if original else current)
        return found

    def _check_weights(self) -> None:
        token = _weight_token(self.layers[0])
        if token != self._weights:
            raise RuntimeError("LoRA A/B weights changed while setting the scale")

    @contextmanager
    def contribution(self, factor: float):
        self.apply(factor)
        try:
            yield
        finally:
            self.restore()


def _weight_token(layer) -> tuple:
    name = next(iter(layer.scaling))
    a_layer = layer.lora_A[name]
    b_layer = layer.lora_B[name]
    a_weight = a_layer.weight
    b_weight = b_layer.weight
    return (
        id(a_weight),
        _first_value(a_weight),
        id(b_weight),
        _first_value(b_weight),
    )


def _first_value(weight):
    data = getattr(weight, "data", weight)
    flat = data.flatten() if hasattr(data, "flatten") else data
    value = flat[0]
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)
