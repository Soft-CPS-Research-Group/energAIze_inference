"""Provisional algorithm interface and explicit installation bindings."""

from typing import List, Literal, Optional

from pydantic import Field, conint, constr, root_validator

from .common import Contract, Name, unique
from .installation import InstallationContext, SignalMeaning


class SignalRequirement(Contract):
    name: Name
    meaning: SignalMeaning


class BundleInterface(Contract):
    schema_version: Literal["bundle-interface.v1-draft"]
    bundle_id: Name
    revision: Name
    manifest_sha256: Optional[constr(strict=True, regex=r"^[a-f0-9]{64}$")] = None
    target_site_id: Name
    agent_index: conint(strict=True, ge=0)
    runtime: Name
    topology_support: Literal["fixed", "dynamic", "unspecified"]
    inputs: List[SignalRequirement]
    outputs: List[SignalRequirement] = Field(..., min_items=1)

    @root_validator(skip_on_failure=True)
    def distinct_names(cls, values):
        unique(values["inputs"], "name", "input name")
        unique(values["outputs"], "name", "output name")
        return values


class Binding(Contract):
    signal_name: Name
    channel_id: Name


class Bindings(Contract):
    revision: Name
    inputs: List[Binding]
    outputs: List[Binding]

    @root_validator(skip_on_failure=True)
    def distinct(cls, values):
        for direction in ("inputs", "outputs"):
            unique(values[direction], "signal_name", f"{direction} binding")
        unique(values["outputs"], "channel_id", "output channel")
        return values


class MessageProfile(Contract):
    profile_id: Literal["percepta.mongo-message.v0"]
    adapter_revision: Name
    # The current producer sends neither a schema version nor an installation revision.
    producer_revision: Optional[Name] = None


class SafetyPolicy(Contract):
    schema_version: Literal["safety-policy.v1-draft"]
    revision: Name
    mode: Literal["off", "observe", "intervene"]
    profile_id: Name
    # Rule implementation/capabilities are validated during candidate preparation.
    rule_ids: List[Name]

    @root_validator(skip_on_failure=True)
    def rules(cls, values):
        rules = values["rule_ids"]
        if len(rules) != len(set(rules)):
            raise ValueError("Duplicate safety rule")
        if values["mode"] != "off" and not rules:
            raise ValueError("Enabled safety requires explicit rules")
        return values


class ConfigurationDraft(Contract):
    schema_version: Literal["configuration.v1-draft"]
    revision: Name
    installation: InstallationContext
    bundle: BundleInterface
    bindings: Bindings
    message: MessageProfile
    safety: SafetyPolicy

    @root_validator(skip_on_failure=True)
    def binding_consistency(cls, values):
        installation, bundle, bindings = values["installation"], values["bundle"], values["bindings"]
        if installation.site_id != bundle.target_site_id:
            raise ValueError("Bundle target_site_id differs from installation site_id")
        channels = {channel.channel_id: channel for channel in installation.channels}
        for direction in ("inputs", "outputs"):
            requirements = {item.name: item for item in getattr(bundle, direction)}
            assigned = {item.signal_name: item for item in getattr(bindings, direction)}
            if requirements.keys() != assigned.keys():
                raise ValueError(f"{direction} bindings must cover exactly the declared signals")
            for name, binding in assigned.items():
                channel = channels.get(binding.channel_id)
                if channel is None:
                    raise ValueError(f"Unknown channel {binding.channel_id} for {name}")
                if channel.meaning != requirements[name].meaning:
                    raise ValueError(f"Signal meaning mismatch for {name}; implicit conversion is forbidden")
                if direction == "outputs" and channel.role != "action":
                    raise ValueError(f"Output {name} must bind an action channel")
                if direction == "inputs" and channel.role == "action":
                    raise ValueError(f"Input {name} cannot bind an action channel")
        # Access, coverage, hashes, runtime capabilities and physical feasibility
        # still require preparation. Schema validity is not activation permission.
        return values


class CompatibilityIssue(Contract):
    code: Name
    path: Name
    explanation: Name
    required_change: Literal["configuration", "bundle", "adapter", "service", "source_data"]


class CompatibilityReport(Contract):
    schema_version: Literal["compatibility.v1-draft"]
    configuration_revision: Name
    baseline_activation_id: Optional[Name] = None
    status: Literal["compatible", "incompatible", "insufficient_information"]
    checks_performed: List[Name]
    issues: List[CompatibilityIssue]

    @root_validator(skip_on_failure=True)
    def coherent(cls, values):
        if values["status"] == "compatible" and (values["issues"] or not values["checks_performed"]):
            raise ValueError("Compatible requires explicit checks and no unresolved issues")
        if values["status"] != "compatible" and not values["issues"]:
            raise ValueError("Unsuccessful compatibility needs a reason")
        return values
