"""Compile boundary contracts into immutable, framework-free execution context."""

from app.version import __version__
from app.core.execution import ExecutionContext, ExecutionIdentity, InputBinding, OutputBinding


def compile_context(config, manifest, candidate_id):
    channels = {c.channel_id: c for c in config.installation.channels}
    inputs = {b.signal_name: channels[b.channel_id] for b in config.bindings.inputs}
    outputs = {b.signal_name: channels[b.channel_id] for b in config.bindings.outputs}
    encoders = manifest.get_encoder_specs(config.bundle.agent_index)
    identity = ExecutionIdentity(
        candidate_id, config.installation.site_id, config.revision, config.installation.revision,
        config.bindings.revision, config.bundle.bundle_id, config.bundle.revision,
        config.bundle.manifest_sha256, config.bundle.agent_index, config.message.profile_id,
        config.message.adapter_revision, config.message.producer_revision, config.safety.revision, __version__,
    )
    return ExecutionContext(identity, tuple(
        InputBinding(r.name, inputs[r.name].channel_id, tuple(inputs[r.name].message_path),
                     r.meaning.value_type, r.meaning.unit, r.meaning.interval_seconds,
                     tuple(e.params["classes"]) if e.type == "OnehotEncoding" else None)
        for r, e in zip(config.bundle.inputs, encoders)
    ), tuple(
        OutputBinding(r.name, outputs[r.name].channel_id, outputs[r.name].equipment_id, r.meaning.unit)
        for r in config.bundle.outputs
    ))
