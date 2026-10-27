"""Evaluate one captured context and runtime; no active store or physical dispatch."""

from dataclasses import asdict
import math

from app.core.errors import ExecutionError, InputError


class CandidateExecution:
    def __init__(self, context, adapter, runtime):
        self.context, self.adapter, self.runtime = context, adapter, runtime

    def evaluate(self, features):
        context, adapter, runtime = self.context, self.adapter, self.runtime
        prepared, evidence = adapter.prepare(features)
        try:
            # ONNX exposes the raw proposal separately from legacy formatting.
            proposal = runtime.propose(prepared)
        except (InputError, ExecutionError):
            raise
        except Exception as exc:
            raise ExecutionError("runtime_failed", "The selected runtime failed", "/runtime") from exc
        if not isinstance(proposal, dict) or set(proposal) != {b.signal_name for b in context.outputs}:
            raise ExecutionError("output_contract_invalid", "Runtime output names differ from the bound outputs", "/runtime/outputs")
        actions = []
        for binding in context.outputs:
            value = proposal[binding.signal_name]
            try:
                valid = type(value) in (int, float) and math.isfinite(value)
            except OverflowError:
                valid = False
            if not valid:
                raise ExecutionError("output_nonfinite", "Runtime output must be a finite number", "/runtime/outputs")
            actions.append({"action_name": binding.signal_name, "channel_id": binding.channel_id,
                            "equipment_id": binding.equipment_id, "unit": binding.unit, "proposed_value": value})
        return {"schema_version": "evaluation.v1-draft", "proposal_only": True,
                "context": asdict(context.identity), "proposed_actions": actions, "input_evidence": evidence}
