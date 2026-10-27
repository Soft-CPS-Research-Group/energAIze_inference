"""Immutable execution bindings compiled before evaluating a request."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ExecutionIdentity:
    candidate_id: str
    site_id: str
    configuration_revision: str
    installation_revision: str
    bindings_revision: str
    bundle_id: str
    bundle_revision: str
    manifest_sha256: str
    agent_index: int
    message_profile: str
    adapter_revision: str
    producer_revision: str | None
    safety_policy_revision: str
    service_version: str


@dataclass(frozen=True)
class InputBinding:
    signal_name: str
    channel_id: str
    path: tuple[str, ...]
    value_type: str
    unit: str
    interval_seconds: float | None
    classes: tuple | None = None


@dataclass(frozen=True)
class OutputBinding:
    signal_name: str
    channel_id: str
    equipment_id: str | None
    unit: str


@dataclass(frozen=True)
class ExecutionContext:
    identity: ExecutionIdentity
    inputs: tuple[InputBinding, ...]
    outputs: tuple[OutputBinding, ...]
