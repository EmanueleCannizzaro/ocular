"""
Document Tracking Service - Prevents duplicate document processing.

This service tracks processed documents by their content hash (SHA-256) to avoid
reprocessing the same document multiple times, regardless of filename.
"""

import hashlib
from datetime import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path

from sqlalchemy.orm import Session
from sqlalchemy import desc

from ocular.database.models import ProcessedDocument


class DocumentTrackingService:
    """Service for tracking processed documents and preventing duplicates."""

    def __init__(self, db: Session):
        self.db = db

    @staticmethod
    def calculate_file_hash(file_path: str) -> str:
        """
        Calculate SHA-256 hash of file content.

        Args:
            file_path: Path to the file

        Returns:
            SHA-256 hash as hex string
        """
        sha256_hash = hashlib.sha256()

        with open(file_path, "rb") as f:
            # Read file in chunks to handle large files efficiently
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)

        return sha256_hash.hexdigest()

    @staticmethod
    def calculate_bytes_hash(file_bytes: bytes) -> str:
        """
        Calculate SHA-256 hash of bytes content.

        Args:
            file_bytes: File content as bytes

        Returns:
            SHA-256 hash as hex string
        """
        return hashlib.sha256(file_bytes).hexdigest()

    def check_duplicate(self, file_hash: str) -> Optional[ProcessedDocument]:
        """
        Check if a document with this hash has been processed before.

        Args:
            file_hash: SHA-256 hash of the document

        Returns:
            ProcessedDocument if duplicate found, None otherwise
        """
        return self.db.query(ProcessedDocument).filter(
            ProcessedDocument.file_hash == file_hash
        ).first()

    def is_duplicate(self, file_path: str) -> bool:
        """
        Check if a file has been processed before (by hash).

        Args:
            file_path: Path to the file to check

        Returns:
            True if duplicate, False otherwise
        """
        file_hash = self.calculate_file_hash(file_path)
        return self.check_duplicate(file_hash) is not None

    def record_processing(
        self,
        file_path: str,
        file_hash: Optional[str] = None,
        provider_used: Optional[str] = None,
        strategy_used: Optional[str] = None,
        extracted_text: Optional[str] = None,
        confidence_score: Optional[float] = None,
        processing_time: Optional[float] = None,
        token_count: Optional[int] = None,
        cost_usd: Optional[float] = None,
        processing_status: str = "completed",
        error_message: Optional[str] = None,
        user_id: Optional[int] = None,
        job_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ProcessedDocument:
        """
        Record a processed document or update existing record.

        Args:
            file_path: Path to the processed file
            file_hash: Pre-calculated hash (optional, will calculate if not provided)
            provider_used: OCR provider name
            strategy_used: Processing strategy
            extracted_text: Extracted text (will store preview)
            confidence_score: OCR confidence (0.0-1.0)
            processing_time: Processing duration in seconds
            token_count: Number of tokens used
            cost_usd: Processing cost in USD
            processing_status: Status (completed, failed, skipped)
            error_message: Error message if failed
            user_id: User who processed the document
            job_id: Batch job ID if part of batch
            metadata: Additional metadata

        Returns:
            ProcessedDocument record
        """
        # Calculate hash if not provided
        if file_hash is None:
            file_hash = self.calculate_file_hash(file_path)

        # Get file info
        path = Path(file_path)
        file_name = path.name
        file_size = path.stat().st_size if path.exists() else None

        # Determine mime type
        mime_type = self._guess_mime_type(file_name)

        # Extract text preview (first 500 chars)
        text_preview = extracted_text[:500] if extracted_text else None

        # Check if document already exists
        existing = self.check_duplicate(file_hash)

        if existing:
            # Update existing record
            existing.last_processed_at = datetime.utcnow()
            existing.processing_count += 1
            existing.provider_used = provider_used
            existing.strategy_used = strategy_used
            existing.extracted_text_preview = text_preview
            existing.confidence_score = confidence_score
            existing.processing_time_seconds = processing_time
            existing.token_count = token_count
            existing.cost_usd = cost_usd
            existing.processing_status = processing_status
            existing.error_message = error_message
            existing.user_id = user_id
            existing.job_id = job_id
            if metadata:
                existing.extra_metadata = metadata

            self.db.commit()
            self.db.refresh(existing)
            return existing
        else:
            # Create new record
            doc = ProcessedDocument(
                file_hash=file_hash,
                file_name=file_name,
                file_size_bytes=file_size,
                mime_type=mime_type,
                first_processed_at=datetime.utcnow(),
                last_processed_at=datetime.utcnow(),
                processing_count=1,
                provider_used=provider_used,
                strategy_used=strategy_used,
                extracted_text_preview=text_preview,
                confidence_score=confidence_score,
                processing_time_seconds=processing_time,
                token_count=token_count,
                cost_usd=cost_usd,
                processing_status=processing_status,
                error_message=error_message,
                user_id=user_id,
                job_id=job_id,
                extra_metadata=metadata
            )

            self.db.add(doc)
            self.db.commit()
            self.db.refresh(doc)
            return doc

    def get_processed_document(self, file_hash: str) -> Optional[ProcessedDocument]:
        """
        Get processed document record by hash.

        Args:
            file_hash: SHA-256 hash of the document

        Returns:
            ProcessedDocument if found, None otherwise
        """
        return self.check_duplicate(file_hash)

    def get_all_processed_documents(
        self,
        user_id: Optional[int] = None,
        job_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[ProcessedDocument]:
        """
        Get list of processed documents with optional filtering.

        Args:
            user_id: Filter by user ID
            job_id: Filter by job ID
            limit: Maximum number of records to return
            offset: Number of records to skip

        Returns:
            List of ProcessedDocument records
        """
        query = self.db.query(ProcessedDocument)

        if user_id is not None:
            query = query.filter(ProcessedDocument.user_id == user_id)

        if job_id is not None:
            query = query.filter(ProcessedDocument.job_id == job_id)

        return query.order_by(desc(ProcessedDocument.last_processed_at))\
            .limit(limit)\
            .offset(offset)\
            .all()

    def get_processing_stats(self, user_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Get processing statistics.

        Args:
            user_id: Filter by user ID (optional)

        Returns:
            Dictionary with statistics
        """
        query = self.db.query(ProcessedDocument)

        if user_id is not None:
            query = query.filter(ProcessedDocument.user_id == user_id)

        all_docs = query.all()

        total_documents = len(all_docs)
        total_processing_count = sum(doc.processing_count for doc in all_docs)
        duplicates_avoided = total_processing_count - total_documents
        total_cost = sum(doc.cost_usd or 0 for doc in all_docs)
        total_tokens = sum(doc.token_count or 0 for doc in all_docs)

        # Count by status
        completed = len([d for d in all_docs if d.processing_status == "completed"])
        failed = len([d for d in all_docs if d.processing_status == "failed"])
        skipped = len([d for d in all_docs if d.processing_status == "skipped"])

        # Average processing time
        processing_times = [d.processing_time_seconds for d in all_docs if d.processing_time_seconds]
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0

        return {
            "total_unique_documents": total_documents,
            "total_processing_count": total_processing_count,
            "duplicates_avoided": duplicates_avoided,
            "duplicate_percentage": (duplicates_avoided / total_processing_count * 100) if total_processing_count > 0 else 0,
            "total_cost_usd": round(total_cost, 2),
            "total_tokens": total_tokens,
            "avg_processing_time_seconds": round(avg_processing_time, 2),
            "status_breakdown": {
                "completed": completed,
                "failed": failed,
                "skipped": skipped
            }
        }

    def delete_processed_document(self, file_hash: str) -> bool:
        """
        Delete a processed document record.

        Args:
            file_hash: SHA-256 hash of the document

        Returns:
            True if deleted, False if not found
        """
        doc = self.check_duplicate(file_hash)
        if doc:
            self.db.delete(doc)
            self.db.commit()
            return True
        return False

    def cleanup_old_records(self, days: int = 90) -> int:
        """
        Delete old processed document records.

        Args:
            days: Delete records older than this many days

        Returns:
            Number of records deleted
        """
        from datetime import timedelta

        cutoff_date = datetime.utcnow() - timedelta(days=days)

        count = self.db.query(ProcessedDocument).filter(
            ProcessedDocument.last_processed_at < cutoff_date
        ).delete()

        self.db.commit()
        return count

    @staticmethod
    def _guess_mime_type(filename: str) -> Optional[str]:
        """Guess MIME type from filename extension."""
        extension_map = {
            '.pdf': 'application/pdf',
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.bmp': 'image/bmp',
            '.tiff': 'image/tiff',
            '.tif': 'image/tiff',
            '.webp': 'image/webp',
            '.gif': 'image/gif'
        }

        ext = Path(filename).suffix.lower()
        return extension_map.get(ext)


# Helper function to get service instance
def get_document_tracking_service(db: Session) -> DocumentTrackingService:
    """
    Get DocumentTrackingService instance.

    Args:
        db: Database session

    Returns:
        DocumentTrackingService instance
    """
    return DocumentTrackingService(db)
