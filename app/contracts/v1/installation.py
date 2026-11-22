"""Reduced installation knowledge delivered by Runtime, without credentials."""

from typing import List, Literal, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import Field, StrictBool, root_validator, validator

from .common import Contract, Name, Number, unique


class SourceReference(Contract):
    system: Name
    reference: Name
    revision: Name


class Bounds(Contract):
    lower: Optional[Number] = None
    upper: Optional[Number] = None

    @root_validator(skip_on_failure=True)
    def ordered(cls, values):
        low, high = values["lower"], values["upper"]
        if low is None and high is None:
            raise ValueError("Specify at least one bound; omit bounds when unknown")
        if low is not None and high is not None and low > high:
            raise ValueError("lower exceeds upper")
        return values


class SignalMeaning(Contract):
    quantity: Name
    unit: Name
    value_type: Literal["number", "string", "boolean", "number_vector"]
    value_semantics: Literal["instantaneous", "state", "interval", "cumulative", "forecast", "command", "unknown"]
    positive_direction: Name
    interval_seconds: Optional[Number] = None

    @validator("interval_seconds")
    def positive_interval(cls, value):
        if value is not None and value <= 0:
            raise ValueError("interval_seconds must be positive")
        return value

    @root_validator(skip_on_failure=True)
    def interval_required(cls, values):
        if values["value_semantics"] in ("interval", "forecast") and values["interval_seconds"] is None:
            raise ValueError("Interval/forecast signals require interval_seconds")
        return values


class TechnicalValue(Contract):
    name: Name
    value: Number
    unit: Name


class Equipment(Contract):
    equipment_id: Name
    kind: Name
    label: Optional[Name] = None
    phases: List[Literal["L1", "L2", "L3"]] = Field(default_factory=list)
    technical_values: List[TechnicalValue] = Field(default_factory=list)

    @root_validator(skip_on_failure=True)
    def distinct(cls, values):
        if len(set(values["phases"])) != len(values["phases"]):
            raise ValueError("Duplicate equipment phase")
        unique(values["technical_values"], "name", "technical parameter")
        return values


class Channel(Contract):
    channel_id: Name
    equipment_id: Optional[Name] = None  # None means a channel owned by this site.
    role: Literal["observation", "forecast", "action"]
    meaning: SignalMeaning
    enabled: StrictBool
    access: Literal["read", "write", "read_write", "unavailable", "unknown"]
    # Exact JSON object keys below features, not a dot-separated expression.
    message_path: Optional[List[Name]] = None
    physical_bounds: Optional[Bounds] = None

    @root_validator(skip_on_failure=True)
    def shape(cls, values):
        path = values["message_path"]
        if path is not None and (not path or path[0] not in ("observations", "forecasts", "community", "entities", "derived_data")):
            raise ValueError("message_path must start at observations, forecasts, community, entities or derived_data")
        if values["role"] == "action" and path is not None:
            raise ValueError("Actions do not read a message_path")
        if values["physical_bounds"] is not None and values["meaning"].value_type not in ("number", "number_vector"):
            raise ValueError("Physical bounds require numeric channels")
        return values


class ConstraintTerm(Contract):
    channel_id: Name
    coefficient: Number


class AggregateLimit(Contract):
    constraint_id: Name
    unit: Name
    terms: List[ConstraintTerm] = Field(..., min_items=1)
    bounds: Bounds

    @root_validator(skip_on_failure=True)
    def distinct_terms(cls, values):
        unique(values["terms"], "channel_id", "constraint channel")
        if any(term.coefficient == 0 for term in values["terms"]):
            raise ValueError("Constraint terms must have nonzero coefficients")
        return values


class InstallationContext(Contract):
    schema_version: Literal["installation.v1-draft"]
    site_id: Name
    revision: Name
    timezone: Name
    sources: List[SourceReference] = Field(..., min_items=1)
    equipment: List[Equipment]
    channels: List[Channel]
    aggregate_limits: List[AggregateLimit] = Field(default_factory=list)

    @validator("timezone")
    def known_timezone(cls, value):
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError("Unknown IANA timezone") from exc
        return value

    @root_validator(skip_on_failure=True)
    def references(cls, values):
        equipment = values["equipment"]
        channels = values["channels"]
        unique(equipment, "equipment_id", "equipment_id")
        unique(channels, "channel_id", "channel_id")
        unique(values["aggregate_limits"], "constraint_id", "constraint_id")
        ids = {item.equipment_id for item in equipment}
        by_channel = {item.channel_id: item for item in channels}
        for channel in channels:
            if channel.equipment_id is not None and channel.equipment_id not in ids:
                raise ValueError(f"Unknown equipment_id for channel {channel.channel_id}")
        for limit in values["aggregate_limits"]:
            for term in limit.terms:
                channel = by_channel.get(term.channel_id)
                if channel is None:
                    raise ValueError(f"Unknown constraint channel {term.channel_id}")
                if channel.meaning.unit != limit.unit or channel.meaning.value_type != "number":
                    raise ValueError(f"Constraint {limit.constraint_id} requires scalar channels in {limit.unit}")
        return values
