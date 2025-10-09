"""
Billing Service with Stripe Integration.

Handles invoicing, payment processing, cost tracking, and budget alerts.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import secrets
from sqlalchemy.orm import Session
from sqlalchemy import func, and_
from fastapi import HTTPException, status

from ocular.database.models import (
    Invoice, InvoiceStatus, PaymentMethod,
    CostTracking, BudgetAlert, BudgetAlertType
)


class StripeService:
    """
    Stripe payment integration service.

    Handles Stripe API interactions for payment processing.
    """

    def __init__(self, stripe_secret_key: str):
        """
        Initialize Stripe service.

        Args:
            stripe_secret_key: Stripe secret API key
        """
        self.stripe_secret_key = stripe_secret_key
        self._stripe = None

    @property
    def stripe(self):
        """Lazy load Stripe library."""
        if self._stripe is None:
            try:
                import stripe
                stripe.api_key = self.stripe_secret_key
                self._stripe = stripe
            except ImportError:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Stripe library not installed"
                )
        return self._stripe

    async def create_customer(self, user_id: int, email: str, name: Optional[str] = None) -> str:
        """
        Create a Stripe customer.

        Args:
            user_id: User ID
            email: Customer email
            name: Customer name

        Returns:
            Stripe customer ID
        """
        customer = self.stripe.Customer.create(
            email=email,
            name=name,
            metadata={"user_id": user_id}
        )
        return customer.id

    async def create_payment_intent(
        self,
        amount: float,
        currency: str = "usd",
        customer_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a Stripe payment intent.

        Args:
            amount: Amount in USD
            currency: Currency code
            customer_id: Stripe customer ID
            metadata: Additional metadata

        Returns:
            Payment intent data
        """
        # Convert to cents
        amount_cents = int(amount * 100)

        payment_intent = self.stripe.PaymentIntent.create(
            amount=amount_cents,
            currency=currency,
            customer=customer_id,
            metadata=metadata or {}
        )

        return {
            "id": payment_intent.id,
            "client_secret": payment_intent.client_secret,
            "amount": amount,
            "currency": currency,
            "status": payment_intent.status
        }

    async def create_invoice(
        self,
        customer_id: str,
        line_items: List[Dict[str, Any]],
        auto_advance: bool = True
    ) -> str:
        """
        Create a Stripe invoice.

        Args:
            customer_id: Stripe customer ID
            line_items: Invoice line items
            auto_advance: Auto-finalize invoice

        Returns:
            Stripe invoice ID
        """
        # Create invoice items
        for item in line_items:
            self.stripe.InvoiceItem.create(
                customer=customer_id,
                description=item.get("description"),
                amount=int(item.get("amount", 0) * 100),  # Convert to cents
                currency="usd"
            )

        # Create invoice
        invoice = self.stripe.Invoice.create(
            customer=customer_id,
            auto_advance=auto_advance
        )

        if auto_advance:
            invoice.finalize_invoice()

        return invoice.id

    async def retrieve_invoice(self, invoice_id: str) -> Dict[str, Any]:
        """
        Retrieve a Stripe invoice.

        Args:
            invoice_id: Stripe invoice ID

        Returns:
            Invoice data
        """
        invoice = self.stripe.Invoice.retrieve(invoice_id)
        return {
            "id": invoice.id,
            "status": invoice.status,
            "total": invoice.total / 100,  # Convert from cents
            "amount_due": invoice.amount_due / 100,
            "amount_paid": invoice.amount_paid / 100,
            "paid": invoice.paid,
            "hosted_invoice_url": invoice.hosted_invoice_url,
            "invoice_pdf": invoice.invoice_pdf
        }


class InvoiceService:
    """Service for invoice management."""

    @staticmethod
    def generate_invoice_number() -> str:
        """
        Generate unique invoice number.

        Returns:
            Invoice number (e.g., INV-2025-001234)
        """
        year = datetime.utcnow().year
        random_number = secrets.randbelow(999999)
        return f"INV-{year}-{random_number:06d}"

    @staticmethod
    def create_invoice(
        db: Session,
        user_id: int,
        period_start: datetime,
        period_end: datetime,
        line_items: List[Dict[str, Any]],
        tax_rate: float = 0.0
    ) -> Invoice:
        """
        Create a new invoice.

        Args:
            db: Database session
            user_id: User ID
            period_start: Billing period start
            period_end: Billing period end
            line_items: Invoice line items
            tax_rate: Tax rate (e.g., 0.10 for 10%)

        Returns:
            Created invoice
        """
        # Calculate totals
        subtotal = sum(item.get("amount", 0.0) for item in line_items)
        tax = subtotal * tax_rate
        total = subtotal + tax

        # Generate invoice number
        invoice_number = InvoiceService.generate_invoice_number()

        # Create invoice
        invoice = Invoice(
            user_id=user_id,
            invoice_number=invoice_number,
            invoice_date=datetime.utcnow(),
            due_date=datetime.utcnow() + timedelta(days=30),
            status=InvoiceStatus.PENDING,
            subtotal=subtotal,
            tax=tax,
            total=total,
            amount_paid=0.0,
            period_start=period_start,
            period_end=period_end,
            line_items=line_items
        )

        db.add(invoice)
        db.commit()
        db.refresh(invoice)

        return invoice

    @staticmethod
    def get_invoice(db: Session, invoice_id: int) -> Optional[Invoice]:
        """Get invoice by ID."""
        return db.query(Invoice).filter(Invoice.id == invoice_id).first()

    @staticmethod
    def list_user_invoices(
        db: Session,
        user_id: int,
        skip: int = 0,
        limit: int = 100,
        status: Optional[InvoiceStatus] = None
    ) -> List[Invoice]:
        """
        List user's invoices.

        Args:
            db: Database session
            user_id: User ID
            skip: Records to skip
            limit: Max records to return
            status: Filter by status

        Returns:
            List of invoices
        """
        query = db.query(Invoice).filter(Invoice.user_id == user_id)

        if status:
            query = query.filter(Invoice.status == status)

        return query.order_by(Invoice.invoice_date.desc()).offset(skip).limit(limit).all()

    @staticmethod
    def mark_as_paid(
        db: Session,
        invoice_id: int,
        payment_method: PaymentMethod,
        stripe_payment_intent_id: Optional[str] = None
    ) -> Invoice:
        """
        Mark invoice as paid.

        Args:
            db: Database session
            invoice_id: Invoice ID
            payment_method: Payment method used
            stripe_payment_intent_id: Stripe payment intent ID

        Returns:
            Updated invoice
        """
        invoice = db.query(Invoice).filter(Invoice.id == invoice_id).first()
        if not invoice:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Invoice not found"
            )

        invoice.status = InvoiceStatus.PAID
        invoice.payment_method = payment_method
        invoice.stripe_payment_intent_id = stripe_payment_intent_id
        invoice.amount_paid = invoice.total
        invoice.paid_at = datetime.utcnow()
        invoice.updated_at = datetime.utcnow()

        db.commit()
        db.refresh(invoice)

        return invoice

    @staticmethod
    def generate_monthly_invoice(db: Session, user_id: int) -> Invoice:
        """
        Generate monthly invoice based on cost tracking.

        Args:
            db: Database session
            user_id: User ID

        Returns:
            Generated invoice
        """
        # Calculate last month's period
        today = datetime.utcnow()
        period_end = datetime(today.year, today.month, 1) - timedelta(days=1)
        period_start = datetime(period_end.year, period_end.month, 1)

        # Get cost records for the period
        costs = db.query(CostTracking).filter(
            and_(
                CostTracking.user_id == user_id,
                CostTracking.period_start >= period_start,
                CostTracking.period_end <= period_end
            )
        ).all()

        # Group by resource type
        line_items = {}
        for cost in costs:
            resource_type = cost.resource_type
            if resource_type not in line_items:
                line_items[resource_type] = {
                    "description": f"{resource_type.title()} Usage",
                    "amount": 0.0,
                    "quantity": 0.0,
                    "unit": cost.usage_unit
                }
            line_items[resource_type]["amount"] += cost.cost_usd
            if cost.usage_amount:
                line_items[resource_type]["quantity"] += cost.usage_amount

        # Convert to list
        line_items_list = list(line_items.values())

        # Create invoice
        return InvoiceService.create_invoice(
            db=db,
            user_id=user_id,
            period_start=period_start,
            period_end=period_end,
            line_items=line_items_list,
            tax_rate=0.0  # TODO: Calculate based on location
        )


class CostTrackingService:
    """Service for cost tracking and monitoring."""

    @staticmethod
    def record_cost(
        db: Session,
        user_id: int,
        resource_type: str,
        cost_usd: float,
        period_start: datetime,
        period_end: datetime,
        deployment_id: Optional[int] = None,
        usage_amount: Optional[float] = None,
        usage_unit: Optional[str] = None,
        provider_name: Optional[str] = None,
        resource_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> CostTracking:
        """
        Record a cost entry.

        Args:
            db: Database session
            user_id: User ID
            resource_type: Resource type (deployment, api_call, storage)
            cost_usd: Cost in USD
            period_start: Period start time
            period_end: Period end time
            deployment_id: Deployment ID (if applicable)
            usage_amount: Usage amount
            usage_unit: Usage unit
            provider_name: Provider name
            resource_name: Resource name
            metadata: Additional metadata

        Returns:
            Cost tracking record
        """
        cost_record = CostTracking(
            user_id=user_id,
            deployment_id=deployment_id,
            resource_type=resource_type,
            cost_usd=cost_usd,
            usage_amount=usage_amount,
            usage_unit=usage_unit,
            provider_name=provider_name,
            resource_name=resource_name,
            period_start=period_start,
            period_end=period_end,
            cost_metadata=metadata
        )

        db.add(cost_record)
        db.commit()
        db.refresh(cost_record)

        return cost_record

    @staticmethod
    def get_user_costs(
        db: Session,
        user_id: int,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        resource_type: Optional[str] = None
    ) -> List[CostTracking]:
        """
        Get user's cost records.

        Args:
            db: Database session
            user_id: User ID
            start_date: Filter start date
            end_date: Filter end date
            resource_type: Filter by resource type

        Returns:
            List of cost records
        """
        query = db.query(CostTracking).filter(CostTracking.user_id == user_id)

        if start_date:
            query = query.filter(CostTracking.period_start >= start_date)
        if end_date:
            query = query.filter(CostTracking.period_end <= end_date)
        if resource_type:
            query = query.filter(CostTracking.resource_type == resource_type)

        return query.order_by(CostTracking.created_at.desc()).all()

    @staticmethod
    def get_total_cost(
        db: Session,
        user_id: int,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> float:
        """
        Get total cost for user.

        Args:
            db: Database session
            user_id: User ID
            start_date: Filter start date
            end_date: Filter end date

        Returns:
            Total cost in USD
        """
        query = db.query(func.sum(CostTracking.cost_usd)).filter(
            CostTracking.user_id == user_id
        )

        if start_date:
            query = query.filter(CostTracking.period_start >= start_date)
        if end_date:
            query = query.filter(CostTracking.period_end <= end_date)

        result = query.scalar()
        return result if result else 0.0

    @staticmethod
    def get_cost_by_resource_type(
        db: Session,
        user_id: int,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, float]:
        """
        Get costs grouped by resource type.

        Args:
            db: Database session
            user_id: User ID
            start_date: Filter start date
            end_date: Filter end date

        Returns:
            Dictionary of resource_type -> total_cost
        """
        query = db.query(
            CostTracking.resource_type,
            func.sum(CostTracking.cost_usd).label("total_cost")
        ).filter(CostTracking.user_id == user_id)

        if start_date:
            query = query.filter(CostTracking.period_start >= start_date)
        if end_date:
            query = query.filter(CostTracking.period_end <= end_date)

        query = query.group_by(CostTracking.resource_type)

        results = query.all()
        return {row.resource_type: float(row.total_cost) for row in results}


class BudgetAlertService:
    """Service for budget alert management."""

    @staticmethod
    def create_alert(
        db: Session,
        user_id: int,
        alert_name: str,
        threshold_amount: float,
        alert_type: BudgetAlertType = BudgetAlertType.THRESHOLD,
        period_type: str = "monthly",
        notify_email: bool = True,
        notify_slack: bool = False,
        slack_webhook_url: Optional[str] = None
    ) -> BudgetAlert:
        """
        Create a budget alert.

        Args:
            db: Database session
            user_id: User ID
            alert_name: Alert name
            threshold_amount: Budget threshold in USD
            alert_type: Alert type
            period_type: Period type (daily, weekly, monthly, yearly)
            notify_email: Send email notifications
            notify_slack: Send Slack notifications
            slack_webhook_url: Slack webhook URL

        Returns:
            Created budget alert
        """
        # Calculate period start/end based on type
        now = datetime.utcnow()
        if period_type == "monthly":
            period_start = datetime(now.year, now.month, 1)
            # Last day of month
            if now.month == 12:
                period_end = datetime(now.year + 1, 1, 1) - timedelta(days=1)
            else:
                period_end = datetime(now.year, now.month + 1, 1) - timedelta(days=1)
        elif period_type == "weekly":
            period_start = now - timedelta(days=now.weekday())
            period_end = period_start + timedelta(days=6)
        elif period_type == "daily":
            period_start = datetime(now.year, now.month, now.day)
            period_end = period_start + timedelta(days=1) - timedelta(seconds=1)
        else:  # yearly
            period_start = datetime(now.year, 1, 1)
            period_end = datetime(now.year, 12, 31, 23, 59, 59)

        alert = BudgetAlert(
            user_id=user_id,
            alert_name=alert_name,
            alert_type=alert_type,
            threshold_amount=threshold_amount,
            current_amount=0.0,
            period_type=period_type,
            period_start=period_start,
            period_end=period_end,
            is_active=True,
            is_triggered=False,
            notify_email=notify_email,
            notify_slack=notify_slack,
            slack_webhook_url=slack_webhook_url
        )

        db.add(alert)
        db.commit()
        db.refresh(alert)

        return alert

    @staticmethod
    def update_alert_amount(db: Session, alert_id: int, current_amount: float) -> BudgetAlert:
        """
        Update alert's current amount.

        Args:
            db: Database session
            alert_id: Alert ID
            current_amount: Current amount

        Returns:
            Updated alert
        """
        alert = db.query(BudgetAlert).filter(BudgetAlert.id == alert_id).first()
        if not alert:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Budget alert not found"
            )

        alert.current_amount = current_amount
        alert.updated_at = datetime.utcnow()

        # Check if should trigger
        if alert.should_trigger:
            alert.is_triggered = True
            alert.last_triggered_at = datetime.utcnow()

        db.commit()
        db.refresh(alert)

        return alert

    @staticmethod
    def check_and_update_alerts(db: Session, user_id: int):
        """
        Check and update all alerts for a user.

        Args:
            db: Database session
            user_id: User ID
        """
        alerts = db.query(BudgetAlert).filter(
            and_(
                BudgetAlert.user_id == user_id,
                BudgetAlert.is_active == True
            )
        ).all()

        for alert in alerts:
            # Get current costs for the period
            total_cost = CostTrackingService.get_total_cost(
                db,
                user_id,
                start_date=alert.period_start,
                end_date=alert.period_end
            )

            # Update alert
            BudgetAlertService.update_alert_amount(db, alert.id, total_cost)

    @staticmethod
    def list_user_alerts(
        db: Session,
        user_id: int,
        is_active: Optional[bool] = None
    ) -> List[BudgetAlert]:
        """
        List user's budget alerts.

        Args:
            db: Database session
            user_id: User ID
            is_active: Filter by active status

        Returns:
            List of budget alerts
        """
        query = db.query(BudgetAlert).filter(BudgetAlert.user_id == user_id)

        if is_active is not None:
            query = query.filter(BudgetAlert.is_active == is_active)

        return query.order_by(BudgetAlert.created_at.desc()).all()

    @staticmethod
    def deactivate_alert(db: Session, alert_id: int, user_id: int) -> BudgetAlert:
        """
        Deactivate a budget alert.

        Args:
            db: Database session
            alert_id: Alert ID
            user_id: User ID (for authorization)

        Returns:
            Updated alert
        """
        alert = db.query(BudgetAlert).filter(
            and_(
                BudgetAlert.id == alert_id,
                BudgetAlert.user_id == user_id
            )
        ).first()

        if not alert:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Budget alert not found"
            )

        alert.is_active = False
        alert.updated_at = datetime.utcnow()

        db.commit()
        db.refresh(alert)

        return alert
