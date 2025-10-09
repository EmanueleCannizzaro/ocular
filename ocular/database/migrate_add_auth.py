"""
Database migration script to add authentication tables.

Adds User, APIKey, and AuditLog tables for authentication and authorization.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sqlalchemy import create_engine
from ocular.database.models import Base, User, APIKey, AuditLog
from ocular.database.session import get_database_url


def migrate():
    """Create authentication tables."""
    database_url = get_database_url()
    engine = create_engine(database_url)

    print("Creating authentication tables...")
    print(f"Database URL: {database_url}")

    # Create only the new tables
    User.__table__.create(engine, checkfirst=True)
    print("✓ Created 'users' table")

    APIKey.__table__.create(engine, checkfirst=True)
    print("✓ Created 'api_keys' table")

    AuditLog.__table__.create(engine, checkfirst=True)
    print("✓ Created 'audit_logs' table")

    print("\nMigration completed successfully!")
    print("\nNew tables added:")
    print("  - users (User authentication and profile)")
    print("  - api_keys (API key management)")
    print("  - audit_logs (Audit trail)")


if __name__ == "__main__":
    migrate()
