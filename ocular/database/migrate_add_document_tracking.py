"""
Database migration to add ProcessedDocument table for document tracking.

This migration adds a new table to track processed documents by content hash,
preventing duplicate processing and providing analytics on duplicate avoidance.
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()


class ProcessedDocument(Base):
    """
    Tracks processed documents to prevent duplicate processing.
    Documents are identified by content hash (SHA-256) instead of filename.
    """
    __tablename__ = "processed_documents"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)

    # Document Identification
    file_hash = Column(String(64), unique=True, nullable=False, index=True)  # SHA-256 hash
    file_name = Column(String(512), nullable=True)  # Original filename (for reference)
    file_size_bytes = Column(Integer, nullable=True)  # File size in bytes
    mime_type = Column(String(100), nullable=True)  # e.g., 'application/pdf', 'image/jpeg'

    # Processing Information
    first_processed_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    last_processed_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    processing_count = Column(Integer, nullable=False, default=1)  # Number of times processed

    # OCR Results (latest)
    provider_used = Column(String(100), nullable=True)  # e.g., 'mistral', 'google_vision'
    strategy_used = Column(String(50), nullable=True)  # e.g., 'single', 'fallback'
    extracted_text_preview = Column(Text, nullable=True)  # First 500 chars of extracted text
    confidence_score = Column(Float, nullable=True)  # OCR confidence score (0.0-1.0)

    # Processing Results
    processing_time_seconds = Column(Float, nullable=True)  # Processing duration
    token_count = Column(Integer, nullable=True)  # Number of tokens used (if applicable)
    cost_usd = Column(Float, nullable=True)  # Processing cost in USD

    # Status
    processing_status = Column(String(50), nullable=False, default="completed")  # completed, failed, skipped
    error_message = Column(Text, nullable=True)  # Error message if processing failed

    # Metadata
    user_id = Column(Integer, nullable=True, index=True)  # User who processed the document
    job_id = Column(String(100), nullable=True, index=True)  # Batch job ID if part of batch
    extra_metadata = Column(JSON, nullable=True)  # Additional metadata

    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)


def run_migration(database_url: str = None):
    """
    Run the migration to add ProcessedDocument table.

    Args:
        database_url: Database connection string (optional)
    """
    if database_url is None:
        # Default to SQLite database in project root
        import os
        from pathlib import Path
        project_root = Path(__file__).parent.parent.parent
        database_url = f"sqlite:///{project_root}/ocular.db"

    print(f"🔄 Running migration: Add ProcessedDocument table")
    print(f"📊 Database: {database_url}")

    # Create engine
    engine = create_engine(database_url)

    # Create table
    print("✅ Creating processed_documents table...")
    Base.metadata.create_all(engine, tables=[ProcessedDocument.__table__])

    print("✅ Migration completed successfully!")
    print("\n📋 New table created:")
    print("   - processed_documents (for document hash tracking)")
    print("\n🎯 Features enabled:")
    print("   - Duplicate document detection by SHA-256 hash")
    print("   - Processing history and analytics")
    print("   - Cost tracking and duplicate avoidance stats")
    print("   - Integration with batch processing jobs")


def rollback_migration(database_url: str = None):
    """
    Rollback the migration (drop ProcessedDocument table).

    Args:
        database_url: Database connection string (optional)
    """
    if database_url is None:
        import os
        from pathlib import Path
        project_root = Path(__file__).parent.parent.parent
        database_url = f"sqlite:///{project_root}/ocular.db"

    print(f"⚠️  Rolling back migration: Remove ProcessedDocument table")
    print(f"📊 Database: {database_url}")

    # Create engine
    engine = create_engine(database_url)

    # Drop table
    print("🗑️  Dropping processed_documents table...")
    Base.metadata.drop_all(engine, tables=[ProcessedDocument.__table__])

    print("✅ Rollback completed!")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "rollback":
        # Rollback migration
        database_url = sys.argv[2] if len(sys.argv) > 2 else None
        rollback_migration(database_url)
    else:
        # Run migration
        database_url = sys.argv[1] if len(sys.argv) > 1 else None
        run_migration(database_url)
