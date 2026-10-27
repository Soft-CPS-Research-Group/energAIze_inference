"""Request contracts for planned administrative routes, not active endpoints."""

from typing import Literal, Optional

from pydantic import Field, root_validator

from .common import Contract, Name


class PreparationRequest(Contract):
    schema_version: Literal["preparation.v1-draft"]
    configuration_revision: Name


class ActivationRequest(Contract):
    schema_version: Literal["activation.v1-draft"]
    candidate_id: Name
    # Required null means "I expect there to be no active configuration".
    expected_activation_id: Optional[Name] = Field(...)


class ContractError(Contract):
    code: Name
    path: Name
    detail: Name


class OperationReceipt(Contract):
    operation_id: Name
    status: Literal["pending", "running", "succeeded", "failed", "uncertain"]
    resource_id: Optional[Name] = None
    error: Optional[ContractError] = None

    @root_validator(skip_on_failure=True)
    def outcome(cls, values):
        if values["status"] == "failed" and values["error"] is None:
            raise ValueError("Failed operation requires an error")
        if values["status"] == "succeeded" and (values["error"] is not None or values["resource_id"] is None):
            raise ValueError("Successful operation requires a resource and no error")
        return values
