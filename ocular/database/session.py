"""
Database session management for Ocular OCR application.
"""

import os
from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from ocular.database.models import Base

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./ocular_deployments.db")

# Handle SQLite-specific configuration
connect_args = {}
poolclass = None

if DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}
    # For in-memory SQLite, use StaticPool
    if ":memory:" in DATABASE_URL:
        poolclass = StaticPool

# Create engine
if poolclass:
    engine = create_engine(
        DATABASE_URL,
        connect_args=connect_args,
        poolclass=poolclass,
        echo=False  # Set to True for SQL query logging
    )
else:
    engine = create_engine(
        DATABASE_URL,
        connect_args=connect_args,
        echo=False
    )

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    """
    Initialize the database by creating all tables.

    This should be called once when the application starts.
    """
    # Ensure database directory exists for SQLite
    if DATABASE_URL.startswith("sqlite") and ":memory:" not in DATABASE_URL:
        db_path = DATABASE_URL.replace("sqlite:///", "")
        db_dir = Path(db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)

    # Create all tables
    Base.metadata.create_all(bind=engine)
    print(f"Database initialized at: {DATABASE_URL}")


def get_db() -> Generator[Session, None, None]:
    """
    Dependency function to get database session.

    Usage in FastAPI:
        @router.get("/deployments")
        async def get_deployments(db: Session = Depends(get_db)):
            return db.query(Deployment).all()

    Yields:
        Session: SQLAlchemy database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_db_session() -> Session:
    """
    Get a new database session (for non-FastAPI usage).

    Returns:
        Session: SQLAlchemy database session

    Note:
        Caller is responsible for closing the session.
    """
    return SessionLocal()


def reset_db() -> None:
    """
    Drop all tables and recreate them.

    WARNING: This will delete all data!
    Use only for development/testing.
    """
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    print("Database reset complete")
