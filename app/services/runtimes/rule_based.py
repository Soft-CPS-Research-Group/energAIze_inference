"""Rule dispatch and wrappers around the existing RBC strategies."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.logging import get_logger
from app.services.preprocessor import AgentPreprocessor
from app.services.runtimes.common import _apply_aliases
from app.services.rbc import (
    BreakerOnlyConfig,
    BreakerOnlyRuntime,
    IchargingBreakerRuntime,
    IchargingRuntimeConfig,
    Rh1HouseConfig,
    Rh1HouseRuntime,
)


class RuleBasedRuntime:
    """Runtime for simple rule-based policies described in JSON."""

    def __init__(
        self,
        index: int,
        config: Dict[str, Any],
        action_names: List[str],
        preprocessor: Optional[AgentPreprocessor] = None,
        feature_aliases: Dict[str, str] | None = None,
    ):
        self.index = index
        self.config = config
        self.action_names = action_names
        self.preprocessor = preprocessor if config.get("use_preprocessor") else None
        self.feature_aliases = feature_aliases or {}
        self.rules = config.get("rules", [])
        default_actions = config.get("default_actions", {})
        if not default_actions and action_names:
            default_actions = {name: 0.0 for name in action_names}
        self.default_actions = default_actions
        self.providers = ["rule_based"]
        self.strategy = config.get("strategy")
        self._icharging_runtime: IchargingBreakerRuntime | None = None
        self._breaker_runtime: BreakerOnlyRuntime | None = None
        self._rh1_runtime: Rh1HouseRuntime | None = None
        if self.strategy in {"breaker_allocation", "icharging_breaker"}:
            icharging_cfg = IchargingRuntimeConfig.from_dict(dict(config))
            self._icharging_runtime = IchargingBreakerRuntime(icharging_cfg)
        elif self.strategy in {"breaker_only", "icharging_breaker_v0"}:
            breaker_cfg = BreakerOnlyConfig.from_dict(dict(config))
            self._breaker_runtime = BreakerOnlyRuntime(breaker_cfg)
        elif self.strategy == "rh1_house_rbc_v1":
            rh1_cfg = Rh1HouseConfig.from_dict(dict(config))
            self._rh1_runtime = Rh1HouseRuntime(rh1_cfg)

    def infer(self, payload: Dict[str, float]) -> Dict[str, float]:
        payload = _apply_aliases(payload, self.feature_aliases)
        raw_payload = payload
        if self.preprocessor:
            features = self.preprocessor.transform(payload)
            raw_payload = {f"f{i}": v for i, v in enumerate(features.tolist())}

        if self._icharging_runtime:
            return self._icharging_runtime.allocate(payload)
        if self._breaker_runtime:
            return self._breaker_runtime.allocate(payload)
        if self._rh1_runtime:
            return self._rh1_runtime.allocate(payload)

        for rule in self.rules:
            conditions = rule.get("if", {})
            if all(raw_payload.get(k) == v for k, v in conditions.items()):
                actions = rule.get("actions", {})
                get_logger().debug(
                    "Rule matched", agent_index=self.index, conditions=conditions
                )
                return {k: float(v) for k, v in actions.items()}

        get_logger().debug("No rule matched", agent_index=self.index)
        return {k: float(v) for k, v in self.default_actions.items()}
