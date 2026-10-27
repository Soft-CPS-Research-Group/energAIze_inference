"""ONNX execution and existing output post-processing."""

from __future__ import annotations

from typing import Dict, List, TYPE_CHECKING

from app.logging import get_logger
from app.services.preprocessor import AgentPreprocessor
from app.services.runtimes.common import _apply_aliases

if TYPE_CHECKING:
    import onnxruntime as ort


def _round_down_one_decimal(value: float) -> float:
    import math

    return math.floor(value * 10.0) / 10.0


class OnnxAgentRuntime:
    """Runtime wrapper around an ONNX model for a single agent."""

    def __init__(
        self,
        index: int,
        session: ort.InferenceSession,
        preprocessor: AgentPreprocessor,
        action_names: List[str],
        feature_aliases: Dict[str, str] | None = None,
        action_bounds: Dict[str, List[float]] | None = None,
    ):
        self.index = index
        self.session = session
        self.preprocessor = preprocessor
        self.action_names = action_names
        self.feature_aliases = feature_aliases or {}
        self.providers = session.get_providers()
        self._low = action_bounds.get("low") if action_bounds else None
        self._high = action_bounds.get("high") if action_bounds else None

    def infer(self, payload: Dict[str, float]) -> Dict[str, float]:
        payload = _apply_aliases(payload, self.feature_aliases)
        features = self.preprocessor.transform(payload)
        log = get_logger()
        try:
            log.debug("Running ONNX inference", agent_index=self.index)
            outputs = self.session.run(None, {self.session.get_inputs()[0].name: features.reshape(1, -1)})
            actions = outputs[0].squeeze(0)
            mapping: Dict[str, float] = {}
            for i, value in enumerate(actions.tolist()):
                key = self.action_names[i] if i < len(self.action_names) else f"action_{i}"
                mapping[key] = self._postprocess_value(value, i)
            return mapping
        except Exception:
            log.exception("onnx.inference_failed", agent_index=self.index)
            return {name: 0.0 for name in self.action_names}

    def _postprocess_value(self, value: float, idx: int) -> float:
        # clamp to bounds if provided
        if self._low and idx < len(self._low):
            value = max(value, float(self._low[idx]))
        if self._high and idx < len(self._high):
            value = min(value, float(self._high[idx]))
        # round down to one decimal place
        value = _round_down_one_decimal(value)
        return float(value)


