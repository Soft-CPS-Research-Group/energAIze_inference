"""ONNX execution and existing output post-processing."""

from __future__ import annotations

from typing import Dict, List, TYPE_CHECKING
import math

import numpy as np

from app.core.errors import ExecutionError, InputError
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
        if len(set(action_names)) != len(action_names):
            raise ValueError("ONNX action names must be unique")
        for bound in (self._low, self._high):
            if bound is not None and (len(bound) != len(action_names) or any(type(v) not in (int, float) or not math.isfinite(v) for v in bound)):
                raise ValueError("ONNX bounds must be finite and match the action width")
        if self._low is not None and self._high is not None and any(lo > hi for lo, hi in zip(self._low, self._high)):
            raise ValueError("ONNX lower bound exceeds upper bound")

    def infer(self, payload: Dict[str, float]) -> Dict[str, float]:
        proposal = self.propose(payload)
        return {name: self._postprocess_value(proposal[name], i) for i, name in enumerate(self.action_names)}

    def propose(self, payload: Dict[str, float]) -> Dict[str, float]:
        """Preserve raw model actions for the versioned execution/safety boundary."""
        payload = _apply_aliases(payload, self.feature_aliases)
        try:
            features = self.preprocessor.transform(payload)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise InputError("encoding_failed", "Required inputs could not be encoded", "/features") from exc
        if features.ndim != 1 or not np.isfinite(features).all():
            raise InputError("encoding_nonfinite", "Encoded inputs must form a finite vector", "/features")
        log = get_logger()
        try:
            log.debug("Running ONNX inference", agent_index=self.index)
            outputs = self.session.run(None, {self.session.get_inputs()[0].name: features.reshape(1, -1)})
        except Exception as exc:
            log.exception("onnx.inference_failed", agent_index=self.index)
            raise ExecutionError("onnx_execution_failed", "ONNX execution failed", "/runtime") from exc
        if len(outputs) != 1 or not isinstance(outputs[0], np.ndarray) or outputs[0].shape != (1, len(self.action_names)):
            raise ExecutionError("output_shape_invalid", "ONNX output does not match declared action width", "/runtime/outputs")
        actions = outputs[0]
        if not np.issubdtype(actions.dtype, np.floating) or not np.isfinite(actions).all():
            raise ExecutionError("output_nonfinite", "ONNX output must contain finite floating-point values", "/runtime/outputs")
        return dict(zip(self.action_names, actions[0].tolist()))

    def _postprocess_value(self, value: float, idx: int) -> float:
        if not math.isfinite(value):
            raise ExecutionError("output_nonfinite", "Action must be finite", "/runtime/outputs")
        # Retain legacy formatting, but re-clamp after rounding so non-decimal
        # and negative lower bounds cannot be violated by flooring.
        if self._low and idx < len(self._low):
            value = max(value, float(self._low[idx]))
        if self._high and idx < len(self._high):
            value = min(value, float(self._high[idx]))
        # round down to one decimal place
        value = _round_down_one_decimal(value)
        if self._low and idx < len(self._low):
            value = max(value, float(self._low[idx]))
        if self._high and idx < len(self._high):
            value = min(value, float(self._high[idx]))
        return float(value)
