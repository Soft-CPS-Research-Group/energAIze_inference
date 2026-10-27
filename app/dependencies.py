"""HTTP dependency accessors; state belongs to the application instance."""

from fastapi import Request

from app.application.service import InferenceService
from app.settings import Settings


def get_service(request: Request) -> InferenceService:
    return request.app.state.inference_service


def get_settings(request: Request) -> Settings:
    return request.app.state.settings
