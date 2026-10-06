"""LoRA contribution changes the forward scalar and then returns to 1.0."""
import pytest

from host_finetune.lora_scale import LoraScale


class _Weight:
    def __init__(self, value):
        self.value = value

    def flatten(self):
        return [self.value]


class _Linear:
    def __init__(self, value):
        self.weight = _Weight(value)


class _Layer:
    def __init__(self, alpha=32, rank=16):
        self.scaling = {"default": alpha / rank}
        self.lora_alpha = {"default": alpha}
        self.r = {"default": rank}
        self.use_rslora = {"default": False}
        self.lora_A = {"default": _Linear(0.25)}
        self.lora_B = {"default": _Linear(-0.5)}
        self.merged = False

    def set_scale(self, adapter, scale):
        if self.use_rslora[adapter]:
            self.scaling[adapter] = scale * self.lora_alpha[adapter] / (self.r[adapter] ** 0.5)
        else:
            self.scaling[adapter] = scale * self.lora_alpha[adapter] / self.r[adapter]


class _Model:
    def __init__(self, layers):
        self._layers = layers

    def modules(self):
        return self._layers


def test_intermediate_scale_restores_the_trained_value_and_leaves_weights():
    layer = _Layer()
    trained = layer.scaling["default"]
    a_id = id(layer.lora_A["default"].weight)
    control = LoraScale(_Model([layer]))
    assert control.factors() == [1.0]
    with control.contribution(0.5):
        assert layer.scaling["default"] == pytest.approx(0.5 * trained)
        assert control.factors() == pytest.approx([0.5])
        assert id(layer.lora_A["default"].weight) == a_id
    assert layer.scaling["default"] == pytest.approx(trained)
    assert control.factors() == pytest.approx([1.0])
    assert layer.lora_A["default"].weight.value == 0.25
    assert layer.lora_B["default"].weight.value == -0.5


def test_a_failed_generation_still_restores_full_strength():
    layer = _Layer()
    control = LoraScale(_Model([layer]))
    with pytest.raises(RuntimeError, match="boom"):
        with control.contribution(0.25):
            raise RuntimeError("boom")
    assert control.factors() == pytest.approx([1.0])


def test_merged_or_missing_lora_is_refused():
    merged = _Layer()
    merged.merged = True
    with pytest.raises(RuntimeError, match="merged"):
        LoraScale(_Model([merged]))
    with pytest.raises(RuntimeError, match="no LoRA"):
        LoraScale(_Model([]))
