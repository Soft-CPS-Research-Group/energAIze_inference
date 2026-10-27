"""Conservative checks for the serving capabilities that exist today."""

import math

from app.preparation.bundle import read_json
from app.preparation.storage import PreparationError
from app.services.preprocessor import build_encoder


def issue(code, path, explanation, required_change="configuration"):
    return dict(code=code, path=path, explanation=explanation, required_change=required_change)


def context_issues(config):
    issues = []
    channels = {c.channel_id: c for c in config.installation.channels}
    used = {b.channel_id for b in config.bindings.inputs + config.bindings.outputs}
    outputs = {b.channel_id for b in config.bindings.outputs}
    for c in config.installation.channels:
        path = f"/installation/channels/{c.channel_id}"
        if c.enabled and c.role == "action" and c.channel_id not in outputs:
            issues.append(issue("uncovered_action", path, "Enabled action has no algorithm output", "bundle"))
        if c.channel_id not in used:
            continue
        if not c.enabled:
            issues.append(issue("channel_disabled", path, "Bound channel is disabled"))
        permitted = {"write", "read_write"} if c.role == "action" else {"read", "read_write"}
        if c.access not in permitted:
            issues.append(issue("access_unknown" if c.access == "unknown" else "access_unavailable", path, "Required channel access is not declared", "source_data"))
        if any(value in ("unknown", "unspecified") for value in (c.meaning.unit, c.meaning.quantity, c.meaning.positive_direction, c.meaning.value_semantics)):
            issues.append(issue("meaning_unknown", path, "Signal meaning is incomplete", "source_data"))
        if c.role != "action" and not c.message_path:
            issues.append(issue("mapping_unknown", path, "Input message path is missing", "source_data"))
        if c.role == "action" and c.physical_bounds is None:
            issues.append(issue("limits_unknown", path, "Action physical limits are missing", "source_data"))
    for limit in config.installation.aggregate_limits:
        for term in limit.terms:
            if term.channel_id not in used:
                issues.append(issue("constraint_input_unbound", f"/installation/aggregate_limits/{limit.constraint_id}", "Constraint references a channel unavailable to this candidate", "bundle"))
    if config.bundle.manifest_sha256 is None:
        issues.append(issue("manifest_hash_unknown", "/bundle/manifest_sha256", "Supply the expected manifest hash", "source_data"))
    if config.bundle.topology_support != "fixed":
        issues.append(issue("topology_unsupported", "/bundle/topology_support", "Current preparation supports fixed interfaces only", "service"))
    if config.message.adapter_revision not in ("legacy-1", "channels-1"):
        issues.append(issue("adapter_unsupported", "/message/adapter_revision", "Supported adapter revisions are legacy-1 and channels-1", "adapter"))
    if config.safety.mode != "off" or config.safety.profile_id != "off.v1" or config.safety.rule_ids:
        issues.append(issue("safety_profile_unsupported", "/safety", "Only explicit off.v1 without rules is implemented; observation/intervention require iteration 5", "service"))
    return issues


def impact(before, after):
    if before is None:
        return {"baseline_configuration_revision": None, "changes": [], "required_change": "initial_configuration"}
    old, new = before.dict(), after.dict()
    changes = []
    for section in ("installation", "bundle", "bindings", "message", "safety"):
        if old[section] != new[section]:
            changes.append(section)
    old_channels = {c.channel_id: c for c in before.installation.channels}
    new_channels = {c.channel_id: c for c in after.installation.channels}
    added, removed = sorted(new_channels.keys() - old_channels.keys()), sorted(old_channels.keys() - new_channels.keys())
    semantic = sorted(k for k in old_channels.keys() & new_channels.keys() if old_channels[k].meaning != new_channels[k].meaning)
    changed_channels = sorted(k for k in old_channels.keys() & new_channels.keys() if old_channels[k] != new_channels[k])
    action_change = any(new_channels[k].role == "action" for k in added) or any(old_channels[k].role == "action" for k in removed)
    requirement = "none" if not changes else "configuration"
    if semantic or action_change or "bundle" in changes:
        requirement = "bundle_review"
    if "message" in changes:
        requirement = "adapter_review"
    return dict(baseline_configuration_revision=before.revision, changes=changes, added_channels=added,
                removed_channels=removed, changed_semantics=semantic, changed_channels=changed_channels,
                constraints_changed=before.installation.aggregate_limits != after.installation.aggregate_limits,
                equipment_changed=before.installation.equipment != after.installation.equipment,
                required_change=requirement)


def check_manifest(config, manifest, snapshot):
    """Reject unsupported/ambiguous declarations before the engine is constructed."""
    def require(ok, code, path, text):
        if not ok:
            raise PreparationError(code, text, path)

    artifacts = manifest.agent.artifacts
    indices = [a.agent_index for a in artifacts]
    require(len(indices) == len(set(indices)) and all(i >= 0 for i in indices), "agent_indices_invalid", "/manifest/agent", "Agent indices must be nonnegative and unique")
    require(config.bundle.agent_index in indices, "agent_missing", "/bundle/agent_index", "Selected agent is absent")
    count = manifest.topology.get("num_agents")
    require(count is None or (type(count) is int and count == len(indices)), "topology_mismatch", "/manifest/topology/num_agents", "Agent count differs from the artifacts")
    site = manifest.metadata.get("site_id")
    require(site is None or site == config.installation.site_id, "manifest_site_mismatch", "/manifest/metadata/site_id", "Manifest targets another site")
    require(manifest.environment.interface in (None, "flat"), "interface_unsupported", "/manifest/environment/interface", "Entity runtime validation is not implemented")
    require(manifest.environment.topology_mode in (None, "fixed") and not manifest.environment.entity_encoding, "topology_unsupported", "/manifest/environment", "Dynamic/entity metadata requires a dedicated preparation adapter")
    artifact = manifest.get_artifact(config.bundle.agent_index)
    runtime = artifact.format or "onnx"
    require(runtime in ("onnx", "rule_based") and runtime == config.bundle.runtime, "runtime_mismatch", "/bundle/runtime", "Runtime is unsupported or differs from the manifest")
    names = manifest.get_observation_names(artifact.agent_index)
    actions = manifest.get_action_names(artifact.agent_index)
    require(names == [r.name for r in config.bundle.inputs] and len(set(names)) == len(names), "input_order_mismatch", "/manifest/environment/observation_names", "Input names/order differ from the declared interface")
    require(actions == [r.name for r in config.bundle.outputs] and len(set(actions)) == len(actions), "output_order_mismatch", "/manifest/environment/action_names", "Output names/order differ from the declared interface")
    require(all(r.meaning.value_type == "number" for r in config.bundle.outputs), "output_type_unsupported", "/bundle/outputs", "Current runtimes return scalar numeric actions")
    require(manifest.get_raw_observation_names(artifact.agent_index) == names, "raw_interface_unsupported", "/manifest/environment/raw_observation_names", "Raw/encoded interface requires a dedicated adapter")
    encoders = manifest.get_encoder_specs(artifact.agent_index)
    require(len(names) == len(encoders), "encoder_count_mismatch", "/manifest/environment/encoders", "Every observation must have exactly one encoder")
    width = 0
    for i, (encoder, requirement) in enumerate(zip(encoders, config.bundle.inputs)):
        path = f"/manifest/environment/encoders/{i}"
        try:
            build_encoder(encoder)
        except (ValueError, TypeError) as exc:
            raise PreparationError("encoder_unsupported", str(exc), path) from exc
        require(requirement.meaning.value_type != "number_vector", "vector_interface_unsupported", path, "Vector width must be defined by a future interface adapter")
        p = encoder.params
        allowed_params = {"NoNormalization": set(), "RemoveFeature": set(), "PeriodicNormalization": {"x_max"},
                          "OnehotEncoding": {"classes"}, "Normalize": {"x_min", "x_max"},
                          "NormalizeWithMissing": {"x_min", "x_max", "missing_value", "default"}}
        require(set(p) <= allowed_params[encoder.type], "encoder_parameters_invalid", path, "Unknown encoder parameters cannot be ignored")
        if encoder.type in ("Normalize", "NormalizeWithMissing", "PeriodicNormalization"):
            numeric = list(p.values())
            require(all(type(x) in (int, float) and math.isfinite(x) for x in numeric), "encoder_parameters_invalid", path, "Encoder parameters must be finite numbers")
        if encoder.type in ("Normalize", "NormalizeWithMissing"):
            require(p.get("x_max", 1) > p.get("x_min", 0), "encoder_range_invalid", path, "Normalization range must be positive")
        if encoder.type == "PeriodicNormalization":
            require(p.get("x_max", 1) > 0, "encoder_range_invalid", path, "Period must be positive")
        size = 1
        if encoder.type == "OnehotEncoding":
            classes = p.get("classes", [])
            require(isinstance(classes, list) and bool(classes) and all(type(x) in (str, int, float, bool) for x in classes)
                    and len(set(classes)) == len(classes), "encoder_classes_invalid", path, "One-hot classes must be a nonempty list of unique scalar categories")
            valid_type = {"number": lambda x: type(x) in (int, float) and math.isfinite(x),
                          "string": lambda x: type(x) is str, "boolean": lambda x: type(x) is bool}.get(requirement.meaning.value_type)
            require(valid_type is not None and all(valid_type(x) for x in classes), "encoder_classes_type", path, "Encoder categories must match the declared input type")
            size = len(classes)
        elif encoder.type == "PeriodicNormalization":
            size = 2
        elif encoder.type == "RemoveFeature":
            size = 0
        if runtime == "onnx":
            require(requirement.meaning.value_type in ("number", "boolean") or encoder.type in ("OnehotEncoding", "RemoveFeature"), "input_type_unsupported", path, "ONNX numeric encoding is required")
        else:
            require(encoder.type == "NoNormalization", "rule_encoder_unsupported", path, "Simple equality rules consume unencoded inputs")
        width += size
    if manifest.environment.encoded_observation_names:
        require(len(manifest.get_encoded_observation_names(artifact.agent_index)) == width, "encoded_names_mismatch", "/manifest/environment/encoded_observation_names", "Encoded names do not match encoder width")
    for declared, actual, field in ((artifact.observation_dimension, width, "observation_dimension"), (artifact.action_dimension, len(actions), "action_dimension")):
        require(declared is None or declared == actual, "dimension_mismatch", f"/manifest/agent/{field}", "Declared dimension differs from the interface")
    cfg = dict(artifact.config)
    if cfg.get("config_path"):
        cfg.update(read_json((snapshot.destination / cfg["config_path"]).read_bytes(), "/bundle/config"))
    require(bool(cfg.get("require_observations_envelope")) == bool(artifact.config.get("require_observations_envelope")), "envelope_override_unsupported", "/bundle/config", "Envelope selection must be declared directly on the artifact")
    # legacy-1 can only prepare bindings whose existing flatten/alias behavior agrees.
    aliases = {}
    if snapshot.alias:
        aliases = read_json((snapshot.destination / snapshot.alias).read_bytes(), "/bundle/aliases")
        aliases = {str(k): str(v) for k, v in aliases.items() if not k.startswith("_")}
    channels = {c.channel_id: c for c in config.installation.channels}
    if config.message.adapter_revision == "channels-1":
        require(not aliases, "alias_binding_conflict", "/bundle/aliases", "channels-1 uses explicit JSON bindings; migrate alias paths to bindings and remove alias overrides")
    for binding in config.bindings.inputs if config.message.adapter_revision == "legacy-1" else []:
        path = list(channels[binding.channel_id].message_path or [])
        require(all("." not in key for key in path), "literal_key_unsupported", "/bindings/inputs", "Legacy flattening cannot distinguish dots inside keys")
        if cfg.get("require_observations_envelope"):
            require(bool(path) and path[0] in {"observations", "forecasts", "community", "timestamp"}, "input_mapping_unsupported", "/bindings/inputs", "This top-level field is not forwarded by the observations-envelope adapter")
        if cfg.get("require_observations_envelope") and path and path[0] == "observations":
            path = path[1:]
        name = ".".join(path)
        require(aliases.get(name, name) == binding.signal_name, "input_mapping_unsupported", "/bindings/inputs", "Configured binding is not implemented by legacy-1 flattening/aliases")
    for binding in config.bindings.outputs if config.message.adapter_revision == "legacy-1" else []:
        require(channels[binding.channel_id].equipment_id == binding.signal_name, "output_mapping_unsupported", "/bindings/outputs", "Current direct-action profile requires action names equal to equipment IDs; other routing needs iteration 4")
    physical = {b.signal_name: channels[b.channel_id].physical_bounds for b in config.bindings.outputs}

    def within_physical(name, low, high):
        bounds = physical[name]
        return bounds is None or ((bounds.lower is None or low >= bounds.lower) and (bounds.upper is None or high <= bounds.upper))
    if runtime == "rule_based":
        policy = read_json((snapshot.destination / artifact.path).read_bytes(), "/bundle/policy")
        require(not cfg.get("strategy") and not policy.get("policy_type"), "strategy_contract_unsupported", "/bundle/policy", "Specialized RBC/exported policy needs its own preparation capability adapter")
        require(set(policy) <= {"default_actions", "rules"}, "policy_fields_unsupported", "/bundle/policy", "Unknown policy fields cannot be silently ignored")
        require(set(cfg) <= {"config_path", "require_observations_envelope", "default_actions", "rules", "use_preprocessor", "strategy"}, "runtime_config_unsupported", "/bundle/config", "Configuration fields require a supported runtime adapter")
        require(not cfg.get("use_preprocessor"), "rule_preprocessor_unsupported", "/bundle/config", "Encoded rule conditions need a dedicated contract")
        defaults = cfg.get("default_actions", policy.get("default_actions", {}))
        rules = cfg.get("rules", policy.get("rules", []))
        require(isinstance(rules, list), "policy_invalid", "/bundle/policy/rules", "Rules must be a list")
        for position, rule in enumerate(rules):
            require(isinstance(rule, dict) and set(rule) <= {"if", "actions"}, "policy_invalid", "/bundle/policy/rules", "Unknown rule structure")
            conditions = rule.get("if", {})
            require(isinstance(conditions, dict) and set(conditions) <= set(names), "condition_unknown", f"/bundle/policy/rules/{position}", "Rule condition is not a declared input")
        for values in [defaults] + [r.get("actions", {}) for r in rules]:
            require(isinstance(values, dict) and set(values) == set(actions), "policy_outputs_invalid", "/bundle/policy", "Every rule/default must explicitly provide every action")
            require(all(type(v) in (int, float) and math.isfinite(v) for v in values.values()), "policy_outputs_invalid", "/bundle/policy", "Action values must be finite numbers")
            require(all(within_physical(k, v, v) for k, v in values.items()), "physical_output_mismatch", "/bundle/policy", "Policy output exceeds installation bounds and no intervention profile is implemented")
    else:
        require(set(cfg) <= {"config_path", "require_observations_envelope"}, "runtime_config_unsupported", "/bundle/config", "Unknown ONNX configuration fields require an adapter")
        bounds = manifest.get_action_bounds(artifact.agent_index)
        require(bool(bounds) and all(len(bounds.get(key, [])) == len(actions) for key in ("low", "high")), "bounds_invalid", "/manifest/environment/action_bounds", "ONNX needs complete output bounds")
        for low, high in zip(bounds["low"], bounds["high"]):
            require(all(type(v) in (int, float) and math.isfinite(v) for v in (low, high)) and low <= high, "bounds_invalid", "/manifest/environment/action_bounds", "Bounds must be finite and ordered")
        require(all(within_physical(name, low, high) for name, low, high in zip(actions, bounds["low"], bounds["high"])), "physical_output_mismatch", "/manifest/environment/action_bounds", "ONNX output bounds exceed the installation bounds")
    return artifact, width, actions
