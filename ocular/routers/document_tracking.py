"""
Document Tracking API Router - Manage processed documents and prevent duplicates.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ocular.database.models import ProcessedDocument
from ocular.database.session import get_db
from ocular.routers import templates
from ocular.services.document_tracking_service import DocumentTrackingService, get_document_tracking_service


# Initialize router
router = APIRouter()


# Pydantic models for request/response
class ProcessedDocumentResponse(BaseModel):
    """Response model for processed document."""
    id: int
    file_hash: str
    file_name: Optional[str]
    file_size_bytes: Optional[int]
    mime_type: Optional[str]
    first_processed_at: str
    last_processed_at: str
    processing_count: int
    provider_used: Optional[str]
    strategy_used: Optional[str]
    extracted_text_preview: Optional[str]
    confidence_score: Optional[float]
    processing_time_seconds: Optional[float]
    token_count: Optional[int]
    cost_usd: Optional[float]
    processing_status: str
    error_message: Optional[str]
    user_id: Optional[int]
    job_id: Optional[str]

    class Config:
        from_attributes = True


class ProcessingStatsResponse(BaseModel):
    """Response model for processing statistics."""
    total_unique_documents: int
    total_processing_count: int
    duplicates_avoided: int
    duplicate_percentage: float
    total_cost_usd: float
    total_tokens: int
    avg_processing_time_seconds: float
    status_breakdown: dict


class DuplicateCheckResponse(BaseModel):
    """Response model for duplicate check."""
    is_duplicate: bool
    existing_record: Optional[ProcessedDocumentResponse]


@router.post("/check-duplicate", response_model=DuplicateCheckResponse)
async def check_duplicate(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Check if a file has been processed before by comparing content hash.

    Args:
        file: File to check
        db: Database session

    Returns:
        DuplicateCheckResponse with duplicate status and existing record if found
    """
    service = get_document_tracking_service(db)

    # Read file content and calculate hash
    file_content = await file.read()
    file_hash = service.calculate_bytes_hash(file_content)

    # Check for duplicate
    existing = service.check_duplicate(file_hash)

    return DuplicateCheckResponse(
        is_duplicate=existing is not None,
        existing_record=ProcessedDocumentResponse.from_orm(existing) if existing else None
    )


@router.get("/processed-documents", response_model=List[ProcessedDocumentResponse])
async def get_processed_documents(
    user_id: Optional[int] = None,
    job_id: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """
    Get list of processed documents.

    Args:
        user_id: Filter by user ID (optional)
        job_id: Filter by job ID (optional)
        limit: Maximum number of records (default: 100)
        offset: Number of records to skip (default: 0)
        db: Database session

    Returns:
        List of ProcessedDocumentResponse
    """
    service = get_document_tracking_service(db)
    documents = service.get_all_processed_documents(
        user_id=user_id,
        job_id=job_id,
        limit=limit,
        offset=offset
    )

    return [ProcessedDocumentResponse.from_orm(doc) for doc in documents]


@router.get("/processed-documents/{file_hash}", response_model=ProcessedDocumentResponse)
async def get_processed_document(
    file_hash: str,
    db: Session = Depends(get_db)
):
    """
    Get processed document by hash.

    Args:
        file_hash: SHA-256 hash of the document
        db: Database session

    Returns:
        ProcessedDocumentResponse

    Raises:
        HTTPException: If document not found
    """
    service = get_document_tracking_service(db)
    document = service.get_processed_document(file_hash)

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with hash {file_hash} not found"
        )

    return ProcessedDocumentResponse.from_orm(document)


@router.get("/stats", response_model=ProcessingStatsResponse)
async def get_processing_stats(
    user_id: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """
    Get processing statistics.

    Args:
        user_id: Filter by user ID (optional)
        db: Database session

    Returns:
        ProcessingStatsResponse with statistics
    """
    service = get_document_tracking_service(db)
    stats = service.get_processing_stats(user_id=user_id)

    return ProcessingStatsResponse(**stats)


@router.delete("/processed-documents/{file_hash}")
async def delete_processed_document(
    file_hash: str,
    db: Session = Depends(get_db)
):
    """
    Delete a processed document record.

    Args:
        file_hash: SHA-256 hash of the document
        db: Database session

    Returns:
        Success message

    Raises:
        HTTPException: If document not found
    """
    service = get_document_tracking_service(db)
    deleted = service.delete_processed_document(file_hash)

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with hash {file_hash} not found"
        )

    return {"message": "Document record deleted successfully", "file_hash": file_hash}


@router.post("/cleanup")
async def cleanup_old_records(
    days: int = 90,
    db: Session = Depends(get_db)
):
    """
    Delete old processed document records.

    Args:
        days: Delete records older than this many days (default: 90)
        db: Database session

    Returns:
        Number of records deleted
    """
    service = get_document_tracking_service(db)
    count = service.cleanup_old_records(days=days)

    return {
        "message": f"Cleanup completed",
        "records_deleted": count,
        "days_threshold": days
    }
