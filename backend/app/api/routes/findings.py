"""Endpoints for editing individual findings."""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Header, Request
from starlette.responses import Response

from backend.app.api.deps import (
    AuthPrincipal,
    DbSession,
    assert_guest_session_owns,
    get_guest_workspace,
    get_optional_principal,
)
from backend.app.schemas.finding import FindingRead, FindingUpdate
from backend.app.services.domain_errors import DomainError, NotFoundError
from backend.app.services.finding_service import delete_finding, finding_to_read, guest_finding_to_read, update_finding
from backend.app.services.guest_analysis_service import delete_guest_finding, update_guest_finding
from backend.app.services.guest_workspace import GuestWorkspace

router = APIRouter(prefix="/findings", tags=["findings"])


@router.patch("/{finding_id}", response_model=FindingRead)
async def patch_finding(
    session: DbSession,
    workspace: Annotated[GuestWorkspace, Depends(get_guest_workspace)],
    principal: Annotated[AuthPrincipal | None, Depends(get_optional_principal)],
    request: Request,
    finding_id: uuid.UUID,
    payload: FindingUpdate,
    x_reviewer: Annotated[str, Header(alias="X-Reviewer")],
) -> FindingRead:
    reviewer = x_reviewer.strip()
    if not reviewer:
        raise DomainError("Header X-Reviewer is required.", status_code=422)
    if principal is not None:
        row = await update_finding(session, finding_id=finding_id, payload=payload, reviewer=reviewer)
        await session.commit()
        await session.refresh(row)
        return finding_to_read(row)
    pair = workspace.get_analysis_for_finding(finding_id)
    if pair is None:
        raise NotFoundError("Finding not found.")
    rec, _ = pair
    assert_guest_session_owns(request, rec.session_id)
    row = update_guest_finding(workspace, finding_id=finding_id, payload=payload, reviewer=reviewer)
    return guest_finding_to_read(row)


@router.delete("/{finding_id}")
async def remove_finding(
    session: DbSession,
    workspace: Annotated[GuestWorkspace, Depends(get_guest_workspace)],
    principal: Annotated[AuthPrincipal | None, Depends(get_optional_principal)],
    request: Request,
    finding_id: uuid.UUID,
    x_reviewer: Annotated[str, Header(alias="X-Reviewer")],
) -> Response:
    reviewer = x_reviewer.strip()
    if not reviewer:
        raise DomainError("Header X-Reviewer is required.", status_code=422)
    if principal is not None:
        await delete_finding(session, finding_id=finding_id, reviewer=reviewer)
        await session.commit()
        return Response(status_code=204)
    pair = workspace.get_analysis_for_finding(finding_id)
    if pair is None:
        raise NotFoundError("Finding not found.")
    rec, _ = pair
    assert_guest_session_owns(request, rec.session_id)
    delete_guest_finding(workspace, finding_id=finding_id, reviewer=reviewer)
    return Response(status_code=204)
