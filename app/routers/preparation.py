"""Authenticated local configuration and candidate API; no activation operation."""

import hmac

from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException, Request, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.contracts.v1.execution import ConfigurationDraft
from app.contracts.v1.lifecycle import OperationReceipt
from app.dependencies import get_settings
from app.preparation.models import PrepareCandidate
from app.preparation.storage import PreparationError

bearer = HTTPBearer(auto_error=False)


def require_admin(config=Depends(get_settings), credentials: HTTPAuthorizationCredentials = Depends(bearer)):
    if config.admin_api_token is None:
        if config.configuration_store_dir is not None:
            raise HTTPException(503, "Administrative authentication is not configured")
        return  # Preserve legacy mode when the new lifecycle is disabled.
    supplied = credentials.credentials if credentials else ""
    if not supplied or not hmac.compare_digest(supplied.encode(), config.admin_api_token.get_secret_value().encode()):
        raise HTTPException(401, "Administrative authentication required", headers={"WWW-Authenticate": "Bearer"})


def manager(request: Request, _=Depends(require_admin)):
    service = getattr(request.app.state, "preparation", None)
    if service is None:
        raise HTTPException(503, "Local candidate preparation is not enabled")
    return service


def translate(action):
    try:
        return action()
    except PreparationError as exc:
        raise HTTPException(exc.status, detail=exc.document()) from exc


router = APIRouter(prefix="/admin/v1", tags=["configuration"])


@router.put("/configurations/{revision}")
def put_configuration(revision: str, body: ConfigurationDraft, response: Response, service=Depends(manager)):
    created = translate(lambda: service.put(revision, body))
    response.status_code = 201 if created else 200
    return {"revision": revision, "stored": True, "activated": False}


@router.get("/configurations/{revision}", response_model=ConfigurationDraft)
def get_configuration(revision: str, service=Depends(manager)):
    return translate(lambda: service.configuration(revision))


@router.post("/candidates", status_code=202, response_model=OperationReceipt)
def prepare_candidate(body: PrepareCandidate, tasks: BackgroundTasks,
                      idempotency_key: str = Header(..., min_length=1, max_length=200), service=Depends(manager)):
    operation_id = translate(lambda: service.submit(idempotency_key, body))
    tasks.add_task(service.prepare, operation_id)
    return service.receipt(operation_id)


@router.get("/operations/{operation_id}", response_model=OperationReceipt)
def get_operation(operation_id: str, service=Depends(manager)):
    return translate(lambda: service.receipt(operation_id))


@router.get("/candidates/{candidate_id}")
def get_candidate(candidate_id: str, service=Depends(manager)):
    return translate(lambda: service.candidate(candidate_id))


@router.get("/status")
def get_status(service=Depends(manager)):
    return {"site_id": service.settings.installation_site_id, **service.store.counts(), "activation_supported": False}
