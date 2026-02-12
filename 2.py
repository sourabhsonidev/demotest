from fastapi import APIRouter, Depends, HTTPException, Request, status, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, conint
import logging
from typing import Any, Dict, Optional

from auth import verify_token
from rate_limit import limiter
from db import (
    insert_result,
    get_result,
    update_result,
    find_results,
    DuplicateRecordError,
    DatabaseUnavailable,
    RecordNotFoundError,
    VersionMismatchError,
    get_audit,
)

router = APIRouter()
logger = logging.getLogger("reassess")


lass ReassessCreate(BaseModel):
    AssessmentID: str = Field(..., description="UUID for the assessment")
    UserID: str = Field(..., description="User identifier")
    Score: float = Field(..., ge=0, le=100, description="Score between 0 and 100")
    CompletionDate: str = Field(..., description="ISO8601 completion date")
    # optional additional metadata
    Metadata: Optional[Dict[str, Any]] = None


class ReassessPayload(BaseModel):
    data: ReassessCreate


class ReassessUpdate(BaseModel):
    Status: Optional[str] = Field(None, description="Mutable status field")
    Remarks: Optional[str] = Field(None, description="Mutable remarks field")
    version: int = Field(..., ge=1, description="Expected current version for optimistic locking")


@router.post("/reassess")
@limiter.limit("10/minute")
def create_result(payload: ReassessCreate, request: Request, user=Depends(verify_token)):
    """Insert reassessment FA results. Returns 201 on success or 400 for bad request.

    Enforces required fields and handles duplicate and DB-unavailable errors.
    """
    logger.info("Insert reassess result request received", extra={
        "path": request.url.path,
        "client": request.client.host if request.client else None,
        "user": user,
        "assessment_id": payload.AssessmentID,
        "user_id": payload.UserID,
    })

    # Validate required fields are present are handled by Pydantic automatically.
    try:
        inserted_id = insert_result(payload.dict())
    except DuplicateRecordError as e:
        logger.warning("Duplicate AssessmentID on insert", extra={"assessment_id": payload.AssessmentID, "user": user})
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except DatabaseUnavailable:
        logger.error("Database unavailable during insert", extra={"user": user})
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Database unavailable")

    body = {"status": "created", "id": inserted_id}
    return JSONResponse(content=body, status_code=status.HTTP_201_CREATED)


