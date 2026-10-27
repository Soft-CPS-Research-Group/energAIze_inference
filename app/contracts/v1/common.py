from typing import Union

from pydantic import BaseModel, StrictInt, confloat, constr

Name = constr(strict=True, min_length=1, regex=r"^\S(?:[\s\S]*\S)?$")
Number = Union[confloat(strict=True, allow_inf_nan=False), StrictInt]


class Contract(BaseModel):
    class Config:
        extra = "forbid"
        allow_mutation = False
        allow_inf_nan = False


def unique(items, field, label):
    keys = [getattr(item, field) for item in items]
    if len(keys) != len(set(keys)):
        raise ValueError(f"Duplicate {label}")
