"""Legacy feature alias behavior shared by execution adapters."""

from typing import Dict

def _apply_aliases(payload: Dict[str, float], aliases: Dict[str, str]) -> Dict[str, float]:
    """Apply feature alias mapping to the incoming payload."""
    if not aliases:
        return payload
    transformed = dict(payload)
    for alias, target in aliases.items():
        if alias in transformed and target not in transformed:
            transformed[target] = transformed.pop(alias)
    return transformed


