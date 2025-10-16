from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Dict


def flatten_payload(data: Mapping[str, Any], sep: str = ".") -> Dict[str, Any]:
    """Flatten nested dictionaries/lists into a single-level dict.

    Nested dict keys are concatenated with ``sep``. Lists/tuples use
    indices in square brackets (e.g. ``key[0]``). Scalars are returned as-is.
    """

    flat: Dict[str, Any] = {}

    def _flatten(current: Any, parent_key: str | None) -> None:
        if isinstance(current, Mapping):
            for key, value in current.items():
                new_key = f"{parent_key}{sep}{key}" if parent_key else str(key)
                _flatten(value, new_key)
        elif isinstance(current, Sequence) and not isinstance(current, (str, bytes, bytearray)):
            if parent_key is None:
                raise ValueError("Cannot flatten a sequence without a parent key")
            for idx, value in enumerate(current):
                new_key = f"{parent_key}[{idx}]"
                _flatten(value, new_key)
        else:
            if parent_key is None:
                raise ValueError("Encountered scalar without a key during flattening")
            flat[parent_key] = current

    _flatten(data, None)
    return flat
