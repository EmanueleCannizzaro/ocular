"""
Database migration script to add version management tables.

This script adds the ModelVersion and VersionDeployment tables to the database.
Run this script once to migrate existing databases.
"""

from sqlalchemy import inspect
from ocular.database.session import engine
from ocular.database.models import ModelVersion, VersionDeployment

def table_exists(table_name: str) -> bool:
    """Check if a table exists in the database."""
    inspector = inspect(engine)
    return table_name in inspector.get_table_names()

def run_migration():
    """Run the migration to add version management tables."""
    print("Starting database migration for version management...")

    # Check existing tables
    existing_tables = set(inspect(engine).get_table_names())
    print(f"Existing tables: {existing_tables}")

    # Tables to create
    new_tables = ['model_versions', 'version_deployments']
    tables_to_create = [t for t in new_tables if t not in existing_tables]

    if not tables_to_create:
        print("✅ All version management tables already exist. No migration needed.")
        return

    print(f"Creating new tables: {tables_to_create}")

    # Create only the new tables
    if 'model_versions' in tables_to_create:
        ModelVersion.__table__.create(engine, checkfirst=True)
        print("✅ Created table: model_versions")

    if 'version_deployments' in tables_to_create:
        VersionDeployment.__table__.create(engine, checkfirst=True)
        print("✅ Created table: version_deployments")

    # Verify tables were created
    new_existing_tables = set(inspect(engine).get_table_names())
    if all(t in new_existing_tables for t in new_tables):
        print("✅ Migration completed successfully!")
        print(f"All tables now: {new_existing_tables}")
    else:
        print("⚠️ Warning: Some tables may not have been created properly")

if __name__ == "__main__":
    run_migration()
