"""
Tests for billing service.

Tests invoice management, cost tracking, and budget alerts.
"""

import pytest
from datetime import datetime, timedelta
from sqlalchemy.orm import Session

from ocular.database.models import (
    User, UserRole, Invoice, InvoiceStatus, PaymentMethod,
    CostTracking, BudgetAlert, BudgetAlertType
)
from ocular.services.billing_service import (
    InvoiceService, CostTrackingService, BudgetAlertService
)


@pytest.fixture
def test_user(db: Session):
    """Create a test user."""
    user = User(
        clerk_user_id="test_clerk_id",
        email="test@example.com",
        role=UserRole.USER,
        is_active=True
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


class TestInvoiceService:
    """Test invoice service."""

    def test_generate_invoice_number(self):
        """Test invoice number generation."""
        invoice_number = InvoiceService.generate_invoice_number()

        assert invoice_number.startswith("INV-")
        assert len(invoice_number) == 15  # INV-YYYY-XXXXXX
        assert invoice_number[4:8].isdigit()  # Year
        assert invoice_number[9:].isdigit()  # Random number

    def test_create_invoice(self, db: Session, test_user: User):
        """Test invoice creation."""
        period_start = datetime.utcnow() - timedelta(days=30)
        period_end = datetime.utcnow()

        line_items = [
            {"description": "Deployment Usage", "amount": 50.00},
            {"description": "API Calls", "amount": 25.00}
        ]

        invoice = InvoiceService.create_invoice(
            db=db,
            user_id=test_user.id,
            period_start=period_start,
            period_end=period_end,
            line_items=line_items,
            tax_rate=0.1
        )

        assert invoice.id is not None
        assert invoice.user_id == test_user.id
        assert invoice.subtotal == 75.00
        assert invoice.tax == 7.50
        assert invoice.total == 82.50
        assert invoice.status == InvoiceStatus.PENDING

    def test_list_user_invoices(self, db: Session, test_user: User):
        """Test listing user invoices."""
        # Create test invoices
        for i in range(3):
            period_start = datetime.utcnow() - timedelta(days=60 - i*30)
            period_end = datetime.utcnow() - timedelta(days=30 - i*30)

            InvoiceService.create_invoice(
                db=db,
                user_id=test_user.id,
                period_start=period_start,
                period_end=period_end,
                line_items=[{"description": f"Test {i}", "amount": 10.00}],
                tax_rate=0.0
            )

        # List invoices
        invoices = InvoiceService.list_user_invoices(db, test_user.id)

        assert len(invoices) == 3
        # Should be ordered by date descending
        assert invoices[0].invoice_date >= invoices[1].invoice_date

    def test_mark_as_paid(self, db: Session, test_user: User):
        """Test marking invoice as paid."""
        period_start = datetime.utcnow() - timedelta(days=30)
        period_end = datetime.utcnow()

        invoice = InvoiceService.create_invoice(
            db=db,
            user_id=test_user.id,
            period_start=period_start,
            period_end=period_end,
            line_items=[{"description": "Test", "amount": 100.00}],
            tax_rate=0.0
        )

        # Mark as paid
        paid_invoice = InvoiceService.mark_as_paid(
            db=db,
            invoice_id=invoice.id,
            payment_method=PaymentMethod.STRIPE,
            stripe_payment_intent_id="pi_test123"
        )

        assert paid_invoice.status == InvoiceStatus.PAID
        assert paid_invoice.payment_method == PaymentMethod.STRIPE
        assert paid_invoice.stripe_payment_intent_id == "pi_test123"
        assert paid_invoice.amount_paid == paid_invoice.total
        assert paid_invoice.paid_at is not None


class TestCostTrackingService:
    """Test cost tracking service."""

    def test_record_cost(self, db: Session, test_user: User):
        """Test recording a cost entry."""
        period_start = datetime.utcnow() - timedelta(hours=1)
        period_end = datetime.utcnow()

        cost_record = CostTrackingService.record_cost(
            db=db,
            user_id=test_user.id,
            resource_type="deployment",
            cost_usd=5.50,
            period_start=period_start,
            period_end=period_end,
            usage_amount=1.0,
            usage_unit="hours",
            provider_name="RunPod",
            resource_name="test-deployment"
        )

        assert cost_record.id is not None
        assert cost_record.user_id == test_user.id
        assert cost_record.cost_usd == 5.50
        assert cost_record.resource_type == "deployment"

    def test_get_total_cost(self, db: Session, test_user: User):
        """Test getting total cost."""
        period_start = datetime.utcnow() - timedelta(days=7)
        period_end = datetime.utcnow()

        # Record multiple costs
        costs = [10.00, 20.00, 15.50]
        for cost in costs:
            CostTrackingService.record_cost(
                db=db,
                user_id=test_user.id,
                resource_type="deployment",
                cost_usd=cost,
                period_start=period_start,
                period_end=period_end
            )

        # Get total
        total = CostTrackingService.get_total_cost(
            db=db,
            user_id=test_user.id,
            start_date=period_start
        )

        assert total == sum(costs)

    def test_get_cost_by_resource_type(self, db: Session, test_user: User):
        """Test getting costs grouped by resource type."""
        period_start = datetime.utcnow() - timedelta(days=7)
        period_end = datetime.utcnow()

        # Record costs for different resource types
        resources = {
            "deployment": [10.00, 20.00],
            "api_call": [5.00, 7.50],
            "storage": [2.00]
        }

        for resource_type, costs in resources.items():
            for cost in costs:
                CostTrackingService.record_cost(
                    db=db,
                    user_id=test_user.id,
                    resource_type=resource_type,
                    cost_usd=cost,
                    period_start=period_start,
                    period_end=period_end
                )

        # Get breakdown
        breakdown = CostTrackingService.get_cost_by_resource_type(
            db=db,
            user_id=test_user.id,
            start_date=period_start
        )

        assert breakdown["deployment"] == 30.00
        assert breakdown["api_call"] == 12.50
        assert breakdown["storage"] == 2.00


class TestBudgetAlertService:
    """Test budget alert service."""

    def test_create_alert(self, db: Session, test_user: User):
        """Test creating a budget alert."""
        alert = BudgetAlertService.create_alert(
            db=db,
            user_id=test_user.id,
            alert_name="Monthly Budget",
            threshold_amount=100.00,
            alert_type=BudgetAlertType.THRESHOLD,
            period_type="monthly"
        )

        assert alert.id is not None
        assert alert.user_id == test_user.id
        assert alert.threshold_amount == 100.00
        assert alert.is_active is True
        assert alert.is_triggered is False

    def test_update_alert_amount(self, db: Session, test_user: User):
        """Test updating alert amount."""
        alert = BudgetAlertService.create_alert(
            db=db,
            user_id=test_user.id,
            alert_name="Test Alert",
            threshold_amount=50.00,
            period_type="monthly"
        )

        # Update with amount below threshold
        updated_alert = BudgetAlertService.update_alert_amount(
            db=db,
            alert_id=alert.id,
            current_amount=30.00
        )

        assert updated_alert.current_amount == 30.00
        assert updated_alert.is_triggered is False

        # Update with amount above threshold
        triggered_alert = BudgetAlertService.update_alert_amount(
            db=db,
            alert_id=alert.id,
            current_amount=60.00
        )

        assert triggered_alert.current_amount == 60.00
        assert triggered_alert.is_triggered is True
        assert triggered_alert.last_triggered_at is not None

    def test_check_and_update_alerts(self, db: Session, test_user: User):
        """Test checking and updating all alerts."""
        # Create alert
        alert = BudgetAlertService.create_alert(
            db=db,
            user_id=test_user.id,
            alert_name="Test Alert",
            threshold_amount=100.00,
            period_type="monthly"
        )

        # Record some costs
        period_start = alert.period_start
        period_end = alert.period_end

        CostTrackingService.record_cost(
            db=db,
            user_id=test_user.id,
            resource_type="deployment",
            cost_usd=120.00,
            period_start=period_start,
            period_end=period_end
        )

        # Check and update alerts
        BudgetAlertService.check_and_update_alerts(db, test_user.id)

        # Verify alert was triggered
        db.refresh(alert)
        assert alert.current_amount == 120.00
        assert alert.is_triggered is True

    def test_list_user_alerts(self, db: Session, test_user: User):
        """Test listing user alerts."""
        # Create multiple alerts
        for i in range(3):
            BudgetAlertService.create_alert(
                db=db,
                user_id=test_user.id,
                alert_name=f"Alert {i}",
                threshold_amount=100.00 * (i + 1),
                period_type="monthly"
            )

        # List all alerts
        alerts = BudgetAlertService.list_user_alerts(db, test_user.id)
        assert len(alerts) == 3

        # List only active alerts
        active_alerts = BudgetAlertService.list_user_alerts(
            db,
            test_user.id,
            is_active=True
        )
        assert len(active_alerts) == 3

    def test_deactivate_alert(self, db: Session, test_user: User):
        """Test deactivating an alert."""
        alert = BudgetAlertService.create_alert(
            db=db,
            user_id=test_user.id,
            alert_name="Test Alert",
            threshold_amount=100.00,
            period_type="monthly"
        )

        # Deactivate
        deactivated = BudgetAlertService.deactivate_alert(
            db=db,
            alert_id=alert.id,
            user_id=test_user.id
        )

        assert deactivated.is_active is False
