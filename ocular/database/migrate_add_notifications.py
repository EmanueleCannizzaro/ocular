#!/usr/bin/env python3
"""
Database migration script to add Notification table.

Run this script to create the notifications table in your database.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import create_engine
from ocular.database.models import Base, Notification


def run_migration(database_url: str = None):
    """Run the migration to add Notification table."""
    if database_url is None:
        import os
        from pathlib import Path

        # Default to project root database
        project_root = Path(__file__).parent.parent.parent
        database_url = f"sqlite:///{project_root}/ocular.db"

    print(f"🔄 Running migration: Add Notification table")
    print(f"📦 Database: {database_url}")

    # Create engine
    engine = create_engine(database_url)

    # Create only the Notification table
    print(f"✨ Creating Notification table...")
    Base.metadata.create_all(engine, tables=[Notification.__table__])

    print("✅ Migration completed successfully!")
    print(f"📊 Notification table created")

    # Show example usage
    print("\n📝 Example Usage:")
    print("="*60)
    print("from ocular.database.session import get_db")
    print("from ocular.database.models import Notification")
    print("from datetime import datetime")
    print("")
    print("# Create notification")
    print("notification = Notification(")
    print("    type='deployment',")
    print("    message='New deployment completed',")
    print("    link='/deployments/123',")
    print("    user_id=1,  # or None for global")
    print("    read=False")
    print(")")
    print("db.add(notification)")
    print("db.commit()")
    print("="*60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Add Notification table migration")
    parser.add_argument(
        "--database-url",
        type=str,
        help="Database URL (default: sqlite:///./ocular.db in project root)",
        default=None
    )

    args = parser.parse_args()
    run_migration(args.database_url)
