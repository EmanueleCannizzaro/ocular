"""
Billing and Cost Management API Router.

Provides endpoints for invoices, cost tracking, and budget alerts.
"""

from typing import List, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field

from ocular.database.session import get_db
from ocular.database.models import (
    User, InvoiceStatus, BudgetAlertType
)
from ocular.services.billing_service import (
    InvoiceService, CostTrackingService, BudgetAlertService, StripeService
)
from ocular.services.rbac_service import AuthDependency


router = APIRouter()


# ============================================================================
# Pydantic Models
# ============================================================================

class InvoiceResponse(BaseModel):
    """Invoice response model."""
    id: int
    user_id: int
    invoice_number: str
    invoice_date: str
    due_date: str
    status: str
    subtotal: float
    tax: float
    total: float
    amount_paid: float
    amount_due: float
    period_start: str
    period_end: str
    line_items: List[dict]
    payment_method: Optional[str] = None
    paid_at: Optional[str] = None
    created_at: str


class CostTrackingResponse(BaseModel):
    """Cost tracking response model."""
    id: int
    user_id: int
    deployment_id: Optional[int] = None
    resource_type: str
    cost_usd: float
    usage_amount: Optional[float] = None
    usage_unit: Optional[str] = None
    provider_name: Optional[str] = None
    resource_name: Optional[str] = None
    period_start: str
    period_end: str
    created_at: str


class BudgetAlertResponse(BaseModel):
    """Budget alert response model."""
    id: int
    user_id: int
    alert_name: str
    alert_type: str
    threshold_amount: float
    current_amount: float
    percentage: float
    period_type: str
    period_start: Optional[str] = None
    period_end: Optional[str] = None
    is_active: bool
    is_triggered: bool
    last_triggered_at: Optional[str] = None
    notify_email: bool
    notify_slack: bool
    created_at: str


class BudgetAlertCreateRequest(BaseModel):
    """Budget alert creation request."""
    alert_name: str = Field(..., min_length=1, max_length=255)
    threshold_amount: float = Field(..., gt=0)
    alert_type: BudgetAlertType = BudgetAlertType.THRESHOLD
    period_type: str = Field(default="monthly", pattern="^(daily|weekly|monthly|yearly)$")
    notify_email: bool = True
    notify_slack: bool = False
    slack_webhook_url: Optional[str] = None


class PaymentIntentRequest(BaseModel):
    """Payment intent creation request."""
    invoice_id: int


class CostSummaryResponse(BaseModel):
    """Cost summary response."""
    total_cost: float
    period_start: str
    period_end: str
    breakdown: dict


# ============================================================================
# Invoice Endpoints
# ============================================================================

@router.get("/invoices", response_model=List[InvoiceResponse])
async def list_invoices(
    current_user: User = Depends(AuthDependency),
    skip: int = 0,
    limit: int = 100,
    status: Optional[InvoiceStatus] = None,
    db: Session = Depends(get_db)
):
    """
    List user's invoices.

    Query parameters:
    - skip: Number of records to skip
    - limit: Maximum number of records to return
    - status: Filter by invoice status
    """
    invoices = InvoiceService.list_user_invoices(
        db,
        current_user.id,
        skip,
        limit,
        status
    )

    return [invoice.to_dict() for invoice in invoices]


@router.get("/invoices/{invoice_id}", response_model=InvoiceResponse)
async def get_invoice(
    invoice_id: int,
    current_user: User = Depends(AuthDependency),
    db: Session = Depends(get_db)
):
    """Get invoice by ID."""
    invoice = InvoiceService.get_invoice(db, invoice_id)

    if not invoice:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Invoice not found"
        )

    # Check ownership
    if invoice.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )

    return invoice.to_dict()


@router.post("/invoices/generate-monthly", response_model=InvoiceResponse)
async def generate_monthly_invoice(
    current_user: User = Depends(AuthDependency),
    db: Session = Depends(get_db)
):
    """Generate monthly invoice based on cost tracking."""
    invoice = InvoiceService.generate_monthly_invoice(db, current_user.id)
    return invoice.to_dict()


@router.post("/invoices/{invoice_id}/payment-intent")
async def create_payment_intent(
    invoice_id: int,
    current_user: User = Depends(AuthDependency),
    db: Session = Depends(get_db)
):
    """Create Stripe payment intent for invoice."""
    invoice = InvoiceService.get_invoice(db, invoice_id)

    if not invoice:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Invoice not found"
        )

    if invoice.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )

    if invoice.status == InvoiceStatus.PAID:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invoice already paid"
        )

    # Get Stripe settings from environment
    import os
    stripe_key = os.getenv("STRIPE_SECRET_KEY")
    if not stripe_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Stripe not configured"
        )

    # Create payment intent
    stripe_service = StripeService(stripe_key)
    payment_intent = await stripe_service.create_payment_intent(
        amount=invoice.total,
        customer_id=current_user.stripe_customer_id,
        metadata={
            "invoice_id": invoice_id,
            "user_id": current_user.id
        }
    )

    # Update invoice with payment intent ID
    invoice.stripe_payment_intent_id = payment_intent["id"]
    db.commit()

    return payment_intent


# ============================================================================
# Cost Tracking Endpoints
# ============================================================================

@router.get("/costs", response_model=List[CostTrackingResponse])
async def list_costs(
    current_user: User = Depends(AuthDependency),
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    resource_type: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    List cost records.

    Query parameters:
    - start_date: Filter by start date
    - end_date: Filter by end date
    - resource_type: Filter by resource type
    """
    costs = CostTrackingService.get_user_costs(
        db,
        current_user.id,
        start_date,
        end_date,
        resource_type
    )

    return [cost.to_dict() for cost in costs]


@router.get("/costs/summary", response_model=CostSummaryResponse)
async def get_cost_summary(
    current_user: User = Depends(AuthDependency),
    period: str = "month",  # day, week, month, year
    db: Session = Depends(get_db)
):
    """
    Get cost summary for a period.

    Query parameters:
    - period: Time period (day, week, month, year)
    """
    # Calculate date range
    now = datetime.utcnow()
    if period == "day":
        start_date = datetime(now.year, now.month, now.day)
        end_date = start_date + timedelta(days=1)
    elif period == "week":
        start_date = now - timedelta(days=now.weekday())
        end_date = start_date + timedelta(days=7)
    elif period == "month":
        start_date = datetime(now.year, now.month, 1)
        if now.month == 12:
            end_date = datetime(now.year + 1, 1, 1)
        else:
            end_date = datetime(now.year, now.month + 1, 1)
    else:  # year
        start_date = datetime(now.year, 1, 1)
        end_date = datetime(now.year + 1, 1, 1)

    # Get total cost
    total_cost = CostTrackingService.get_total_cost(
        db,
        current_user.id,
        start_date,
        end_date
    )

    # Get breakdown by resource type
    breakdown = CostTrackingService.get_cost_by_resource_type(
        db,
        current_user.id,
        start_date,
        end_date
    )

    return {
        "total_cost": total_cost,
        "period_start": start_date.isoformat(),
        "period_end": end_date.isoformat(),
        "breakdown": breakdown
    }


@router.get("/costs/current-month")
async def get_current_month_cost(
    current_user: User = Depends(AuthDependency),
    db: Session = Depends(get_db)
):
    """Get current month's total cost."""
    now = datetime.utcnow()
    start_date = datetime(now.year, now.month, 1)

    total_cost = CostTrackingService.get_total_cost(
        db,
        current_user.id,
        start_date=start_date
    )

    breakdown = CostTrackingService.get_cost_by_resource_type(
        db,
        current_user.id,
        start_date=start_date
    )

    return {
        "total_cost": round(total_cost, 2),
        "breakdown": {k: round(v, 2) for k, v in breakdown.items()},
        "month": now.strftime("%B %Y")
    }


# ============================================================================
# Budget Alert Endpoints
# ============================================================================

@router.post("/budget-alerts", response_model=BudgetAlertResponse, status_code=status.HTTP_201_CREATED)
async def create_budget_alert(
    request: BudgetAlertCreateRequest,
    current_user: User = Depends(AuthDependency),
    db: Session = Depends(get_db)
):
    """Create a budget alert."""
    alert = BudgetAlertService.create_alert(
        db=db,
        user_id=current_user.id,
        alert_name=request.alert_name,
        threshold_amount=request.threshold_amount,
        alert_type=request.alert_type,
        period_type=request.period_type,
        notify_email=request.notify_email,
        notify_slack=request.notify_slack,
        slack_webhook_url=request.slack_webhook_url
    )

    return alert.to_dict()


@router.get("/budget-alerts", response_model=List[BudgetAlertResponse])
async def list_budget_alerts(
    current_user: User = Depends(AuthDependency),
    is_active: Optional[bool] = None,
    db: Session = Depends(get_db)
):
    """
    List budget alerts.

    Query parameters:
    - is_active: Filter by active status
    """
    alerts = BudgetAlertService.list_user_alerts(
        db,
        current_user.id,
        is_active
    )

    return [alert.to_dict() for alert in alerts]


@router.post("/budget-alerts/{alert_id}/deactivate", response_model=BudgetAlertResponse)
async def deactivate_budget_alert(
    alert_id: int,
    current_user: User = Depends(AuthDependency),
    db: Session = Depends(get_db)
):
    """Deactivate a budget alert."""
    alert = BudgetAlertService.deactivate_alert(db, alert_id, current_user.id)
    return alert.to_dict()


@router.post("/budget-alerts/check")
async def check_budget_alerts(
    current_user: User = Depends(AuthDependency),
    db: Session = Depends(get_db)
):
    """Check and update all budget alerts."""
    BudgetAlertService.check_and_update_alerts(db, current_user.id)

    # Return updated alerts
    alerts = BudgetAlertService.list_user_alerts(db, current_user.id, is_active=True)
    triggered_alerts = [a for a in alerts if a.is_triggered]

    return {
        "total_alerts": len(alerts),
        "triggered_alerts": len(triggered_alerts),
        "alerts": [a.to_dict() for a in triggered_alerts]
    }


# ============================================================================
# Payment History
# ============================================================================

@router.get("/payments/history")
async def get_payment_history(
    current_user: User = Depends(AuthDependency),
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """Get payment history."""
    # Get paid invoices
    invoices = InvoiceService.list_user_invoices(
        db,
        current_user.id,
        skip=0,
        limit=limit,
        status=InvoiceStatus.PAID
    )

    return {
        "total_payments": len(invoices),
        "total_amount": sum(inv.amount_paid for inv in invoices),
        "payments": [
            {
                "invoice_number": inv.invoice_number,
                "amount": inv.amount_paid,
                "paid_at": inv.paid_at.isoformat() if inv.paid_at else None,
                "payment_method": inv.payment_method.value if inv.payment_method else None
            }
            for inv in invoices
        ]
    }
