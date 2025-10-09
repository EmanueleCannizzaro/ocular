"""
Database migration script to add billing tables.

Adds Invoice, CostTracking, and BudgetAlert tables for billing and cost management.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sqlalchemy import create_engine
from ocular.database.models import Invoice, CostTracking, BudgetAlert
from ocular.database.session import get_database_url


def migrate():
    """Create billing tables."""
    database_url = get_database_url()
    engine = create_engine(database_url)

    print("Creating billing tables...")
    print(f"Database URL: {database_url}")

    # Create only the new tables
    Invoice.__table__.create(engine, checkfirst=True)
    print("✓ Created 'invoices' table")

    CostTracking.__table__.create(engine, checkfirst=True)
    print("✓ Created 'cost_tracking' table")

    BudgetAlert.__table__.create(engine, checkfirst=True)
    print("✓ Created 'budget_alerts' table")

    print("\nMigration completed successfully!")
    print("\nNew tables added:")
    print("  - invoices (Invoice management and payment tracking)")
    print("  - cost_tracking (Real-time cost monitoring)")
    print("  - budget_alerts (Budget threshold monitoring)")


if __name__ == "__main__":
    migrate()
