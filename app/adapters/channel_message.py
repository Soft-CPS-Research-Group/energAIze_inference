"""Provisional Percepta/Mongo adapter using exact, explicitly bound JSON paths.

Only selected signals become features. There is no recursive flattening, alias
fallback, equipment discovery, inferred freshness threshold or implicit zero.
"""

from datetime import datetime, timezone
import math

from app.core.errors import InputError
from app.core.execution import ExecutionContext

PROFILE = "percepta.mongo-message.v0"
REVISION = "channels-1"
# Same-dimension scale conversions only. Energy/power conversion needs an
# explicit integration/averaging contract and is deliberately not guessed.
UNITS = {"W": ("power", 1), "kW": ("power", 1000), "MW": ("power", 1000000),
         "Wh": ("energy", 1), "kWh": ("energy", 1000), "MWh": ("energy", 1000000)}


def pointer(parts):
    return "/features/" + "/".join(str(p).replace("~", "~0").replace("/", "~1") for p in parts)


def finite_number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def read_path(features, path):
    value = features
    descriptor = None
    for index, key in enumerate(path):
        if isinstance(value, dict) and key in ("values", "final_values") and key in value:
            if not isinstance(value[key], list):
                raise InputError("forecast_invalid", "Forecast values must be an array", pointer(path[:index + 1]))
            descriptor = value if key == "values" else {**value, "values": value[key]}
        if isinstance(value, dict) and key in value:
            value = value[key]
        elif isinstance(value, list) and key.isascii() and key.isdecimal() and len(key) <= len(str(len(value))) and str(int(key)) == key and int(key) < len(value):
            value = value[int(key)]
        else:
            raise InputError("input_missing", "Required bound input is absent", pointer(path[:index + 1]))
    return value, descriptor


def timestamp_evidence(features):
    if "timestamp" not in features:
        return {"timestamp": None, "freshness": "unknown"}
    value = features["timestamp"]
    try:
        if isinstance(value, dict) and set(value) == {"$date"}:
            value = value["$date"]
            if isinstance(value, dict) and set(value) == {"$numberLong"}:
                value = int(value["$numberLong"])
            if finite_number(value):
                value = datetime.fromtimestamp(value / 1000, timezone.utc).isoformat()
        if not isinstance(value, str):
            raise ValueError("Expected a timestamp string or Mongo date")
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("Timestamp must include timezone")
        return {"timestamp": parsed.astimezone(timezone.utc).isoformat(), "freshness": "not_assessed"}
    except (TypeError, ValueError, OverflowError, OSError) as exc:
        raise InputError("timestamp_invalid", "Expected an ISO timestamp with timezone or Mongo $date", "/features/timestamp") from exc


class ChannelMessageAdapter:
    def __init__(self, context: ExecutionContext):
        if context.identity.message_profile != PROFILE or context.identity.adapter_revision != REVISION:
            raise ValueError("Unsupported message profile/adapter revision")
        self.context = context

    def prepare(self, features):
        if not isinstance(features, dict):
            raise InputError("message_invalid", "Expected a message object", "/features")
        time = timestamp_evidence(features)
        prepared, evidence = {}, []
        for binding in self.context.inputs:
            value, descriptor = read_path(features, binding.path)
            path = pointer(binding.path)
            valid = {
                "number": finite_number,
                "boolean": lambda x: type(x) is bool,
                "string": lambda x: type(x) is str,
            }.get(binding.value_type)
            if valid is None or not valid(value):
                raise InputError("input_type_invalid", f"Expected finite {binding.value_type} for channel {binding.channel_id}", path)
            source_unit = binding.unit
            source_interval = binding.interval_seconds
            interval_origin = "configuration" if source_interval is not None else "unknown"
            origin = "configuration"
            converted = False
            if descriptor is not None:
                if "measurement_unit" in descriptor:
                    origin = "message"
                    source_unit = descriptor["measurement_unit"]
                    if not isinstance(source_unit, str) or not source_unit:
                        raise InputError("unit_invalid", "Invalid measurement_unit", path)
                if "frequency_seconds" in descriptor:
                    interval_origin = "message"
                    source_interval = descriptor["frequency_seconds"]
                    if not finite_number(source_interval) or source_interval <= 0:
                        raise InputError("interval_invalid", "frequency_seconds must be positive and finite", path)
                    if binding.interval_seconds is not None and source_interval != binding.interval_seconds:
                        raise InputError("interval_mismatch", "Message frequency differs from the channel contract", path)
                if "horizon_seconds" in descriptor:
                    horizon = descriptor["horizon_seconds"]
                    if not finite_number(horizon) or horizon <= 0:
                        raise InputError("horizon_invalid", "Forecast horizon must be positive and finite", path)
                    # A declared frequency is evidence; the configured interval
                    # alone cannot prove the producer's forecast horizon.
                    if "frequency_seconds" in descriptor and not math.isclose(horizon, source_interval * len(descriptor["values"]), rel_tol=1e-9):
                        raise InputError("horizon_mismatch", "Horizon differs from frequency times forecast length", path)
            if source_unit != binding.unit:
                source, target = UNITS.get(source_unit), UNITS.get(binding.unit)
                if binding.value_type != "number" or not source or not target or source[0] != target[0]:
                    raise InputError("unit_mismatch", "No supported conversion between message and channel units", path)
                value = value * (source[1] / target[1])
                converted = True
                if not finite_number(value):
                    raise InputError("input_nonfinite", "Unit conversion produced a nonfinite value", path)
            if binding.classes is not None and not any(type(value) is type(item) and value == item for item in binding.classes):
                # JSON integers and floats represent the same numerical class.
                if not (finite_number(value) and any(finite_number(item) and value == item for item in binding.classes)):
                    raise InputError("category_unknown", "Input is outside the declared encoder classes", path)
            prepared[binding.signal_name] = value
            evidence.append({"signal_name": binding.signal_name, "channel_id": binding.channel_id,
                             "source_unit": source_unit, "unit": binding.unit, "unit_source": origin,
                             "interval_seconds": source_interval, "interval_source": interval_origin, "converted": converted})
        return prepared, {**time, "inputs": evidence}
