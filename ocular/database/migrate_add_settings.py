#!/usr/bin/env python3
"""
Database migration script to add Settings table.

Run this script to create the settings table in your database.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import create_engine
from ocular.database.models import Base, Settings


def run_migration(database_url: str = None):
    """Run the migration to add Settings table."""
    if database_url is None:
        import os
        from pathlib import Path

        # Default to project root database
        project_root = Path(__file__).parent.parent.parent
        database_url = f"sqlite:///{project_root}/ocular.db"

    print(f"🔄 Running migration: Add Settings table")
    print(f"📦 Database: {database_url}")

    # Create engine
    engine = create_engine(database_url)

    # Create only the Settings table
    print(f"✨ Creating Settings table...")
    Base.metadata.create_all(engine, tables=[Settings.__table__])

    print("✅ Migration completed successfully!")
    print(f"📊 Settings table created")

    # Show example usage
    print("\n📝 Example Usage:")
    print("="*60)
    print("from ocular.database.database import get_db")
    print("from ocular.database.models import Settings")
    print("from datetime import datetime")
    print("import json")
    print("")
    print("# Create settings")
    print("settings = Settings(")
    print("    category='global',")
    print("    settings_json=json.dumps({'app_title': 'Ocular OCR'})")
    print(")")
    print("db.add(settings)")
    print("db.commit()")
    print("="*60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Add Settings table migration")
    parser.add_argument(
        "--database-url",
        type=str,
        help="Database URL (default: sqlite:///./ocular.db in project root)",
        default=None
    )

    args = parser.parse_args()
    run_migration(args.database_url)
