from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from app.utils.manifest import EncoderSpec


class Encoder:
    def transform(self, value):  # noqa: ANN001
        raise NotImplementedError


class NoNormalization(Encoder):
    def transform(self, value):
        return value


class PeriodicNormalization(Encoder):
    def __init__(self, x_max: float):
        self.x_max = x_max

    def transform(self, value):
        value = 2 * np.pi * value / self.x_max
        return np.array([np.sin(value), np.cos(value)], dtype=np.float32)


class OnehotEncoding(Encoder):
    def __init__(self, classes: List):
        self.classes = classes
        self.class_to_index = {cls: idx for idx, cls in enumerate(classes)}

    def transform(self, value):
        vector = np.zeros(len(self.classes), dtype=np.float32)
        idx = self.class_to_index.get(value)
        if idx is not None:
            vector[idx] = 1.0
        return vector


class Normalize(Encoder):
    def __init__(self, x_min: float, x_max: float):
        self.x_min = x_min
        self.x_max = x_max

    def transform(self, value):
        if self.x_min == self.x_max:
            return 0.0
        return float((value - self.x_min) / (self.x_max - self.x_min))


class NormalizeWithMissing(Normalize):
    def __init__(self, x_min: float, x_max: float, missing_value: float = -0.1, default: float = 0.0):
        super().__init__(x_min, x_max)
        self.missing_value = missing_value
        self.default = default

    def transform(self, value):
        if value == self.missing_value:
            return self.default
        return super().transform(value)


class RemoveFeature(Encoder):
    def transform(self, value):
        return np.array([], dtype=np.float32)


ENCODER_FACTORIES = {
    "NoNormalization": lambda spec: NoNormalization(),
    "PeriodicNormalization": lambda spec: PeriodicNormalization(spec.params.get("x_max", 1)),
    "OnehotEncoding": lambda spec: OnehotEncoding(spec.params.get("classes", [])),
    "Normalize": lambda spec: Normalize(spec.params.get("x_min", 0), spec.params.get("x_max", 1)),
    "NormalizeWithMissing": lambda spec: NormalizeWithMissing(
        spec.params.get("x_min", 0),
        spec.params.get("x_max", 1),
        spec.params.get("missing_value", -0.1),
        spec.params.get("default", 0.0),
    ),
    "RemoveFeature": lambda spec: RemoveFeature(),
}


@dataclass
class AgentPreprocessor:
    observation_names: List[str]
    encoders: List[Encoder]

    def transform(self, payload: Dict[str, float]) -> np.ndarray:
        transformed_parts: List[np.ndarray] = []
        for name, encoder in zip(self.observation_names, self.encoders):
            if name not in payload:
                raise KeyError(f"Missing observation '{name}' in payload")
            value = payload[name]
            result = encoder.transform(value)
            if isinstance(result, np.ndarray):
                transformed_parts.append(result.astype(np.float32))
            else:
                transformed_parts.append(np.array([result], dtype=np.float32))
        if not transformed_parts:
            return np.array([], dtype=np.float32)
        return np.concatenate(transformed_parts, axis=0)


def build_encoder(spec: EncoderSpec) -> Encoder:
    factory = ENCODER_FACTORIES.get(spec.type)
    if factory is None:
        raise ValueError(f"Unsupported encoder type: {spec.type}")
    return factory(spec)
