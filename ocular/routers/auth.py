"""
Authentication and Authorization API Router.

Provides endpoints for user management, API keys, and audit logs.
"""

from typing import List, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field

from ocular.database.session import get_db
from ocular.database.models import User, UserRole, AuditAction
from ocular.services.auth_service import (
    UserService, APIKeyService, AuditLogService
)
from ocular.services.rbac_service import (
    RBACService, AuthDependency, require_admin
)


router = APIRouter()


# ============================================================================
# Pydantic Models
# ============================================================================

class UserResponse(BaseModel):
    """User response model."""
    id: int
    clerk_user_id: str
    email: str
    username: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    avatar_url: Optional[str] = None
    role: str
    organization_id: Optional[str] = None
    organization_name: Optional[str] = None
    is_active: bool
    is_verified: bool
    subscription_tier: str
    quotas: dict
    created_at: str
    last_login_at: Optional[str] = None


class UserUpdateRequest(BaseModel):
    """User update request model."""
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    username: Optional[str] = None


class RoleUpdateRequest(BaseModel):
    """Role update request model."""
    role: UserRole


class QuotaUpdateRequest(BaseModel):
    """Quota update request model."""
    api_calls_quota: Optional[int] = Field(None, ge=0)
    deployment_quota: Optional[int] = Field(None, ge=0)
    storage_quota_gb: Optional[float] = Field(None, ge=0.0)


class APIKeyCreateRequest(BaseModel):
    """API key creation request model."""
    key_name: str = Field(..., min_length=1, max_length=255)
    scopes: Optional[List[str]] = Field(default=None)
    expires_in_days: Optional[int] = Field(None, ge=1, le=365)
    rate_limit_per_minute: Optional[int] = Field(None, ge=1)
    rate_limit_per_hour: Optional[int] = Field(None, ge=1)


class APIKeyResponse(BaseModel):
    """API key response model."""
    id: int
    user_id: int
    key_name: str
    key_prefix: str
    scopes: List[str]
    status: str
    expires_at: Optional[str] = None
    last_used_at: Optional[str] = None
    usage_count: int
    rate_limit_per_minute: Optional[int] = None
    rate_limit_per_hour: Optional[int] = None
    created_at: str
    revoked_at: Optional[str] = None


class APIKeyCreateResponse(BaseModel):
    """API key creation response with full key."""
    api_key: APIKeyResponse
    full_key: str
    warning: str = "Store this key securely. It will not be shown again."


class AuditLogResponse(BaseModel):
    """Audit log response model."""
    id: int
    user_id: Optional[int] = None
    action: str
    resource_type: str
    resource_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    request_method: Optional[str] = None
    request_path: Optional[str] = None
    description: Optional[str] = None
    changes: Optional[dict] = None
    metadata: Optional[dict] = None
    status: str
    error_message: Optional[str] = None
    created_at: str


# ============================================================================
# User Endpoints
# ============================================================================

@router.get("/me", response_model=UserResponse)
async def get_current_user(
    current_user: User = Depends(AuthDependency),
    db: Session = Depends(get_db)
):
    """Get current authenticated user's information."""
    return current_user.to_dict()


@router.put("/me", response_model=UserResponse)
async def update_current_user(
    request: UserUpdateRequest,
    current_user: User = Depends(AuthDependency),
    db: Session = Depends(get_db)
):
    """Update current user's profile information."""
    if request.first_name is not None:
        current_user.first_name = request.first_name
    if request.last_name is not None:
        current_user.last_name = request.last_name
    if request.username is not None:
        # Check if username is already taken
        existing = db.query(User).filter(
            User.username == request.username,
            User.id != current_user.id
        ).first()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already taken"
            )
        current_user.username = request.username

    current_user.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(current_user)

    # Log action
    AuditLogService.log_action(
        db=db,
        user_id=current_user.id,
        action=AuditAction.UPDATE,
        resource_type="user",
        resource_id=str(current_user.id),
        description="Updated profile information"
    )

    return current_user.to_dict()


@router.get("/users", response_model=List[UserResponse])
@require_admin
async def list_users(
    current_user: User = Depends(AuthDependency),
    skip: int = 0,
    limit: int = 100,
    role: Optional[UserRole] = None,
    is_active: Optional[bool] = None,
    db: Session = Depends(get_db)
):
    """
    List all users (Admin only).

    Query parameters:
    - skip: Number of records to skip
    - limit: Maximum number of records to return
    - role: Filter by role
    - is_active: Filter by active status
    """
    users = UserService.list_users(db, skip, limit, role, is_active)
    return [user.to_dict() for user in users]


@router.get("/users/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: int,
    current_user: User = Depends(AuthDependency),
    db: Session = Depends(get_db)
):
    """
    Get user by ID.

    Users can only view their own profile, admins can view any user.
    """
    if not RBACService.can_access_resource(current_user, user_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )

    user = UserService.get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    return user.to_dict()


@router.put("/users/{user_id}/role", response_model=UserResponse)
@require_admin
async def update_user_role(
    user_id: int,
    request: RoleUpdateRequest,
    current_user: User = Depends(AuthDependency),
    db: Session = Depends(get_db)
):
    """Update user's role (Admin only)."""
    user = UserService.update_user_role(db, user_id, request.role)

    # Log action
    AuditLogService.log_action(
        db=db,
        user_id=current_user.id,
        action=AuditAction.UPDATE,
        resource_type="user",
        resource_id=str(user_id),
        description=f"Updated user role to {request.role.value}",
        changes={"role": request.role.value}
    )

    return user.to_dict()


@router.put("/users/{user_id}/quotas", response_model=UserResponse)
@require_admin
async def update_user_quotas(
    user_id: int,
    request: QuotaUpdateRequest,
    current_user: User = Depends(AuthDependency),
    db: Session = Depends(get_db)
):
    """Update user's quotas (Admin only)."""
    user = UserService.update_user_quotas(
        db,
        user_id,
        request.api_calls_quota,
        request.deployment_quota,
        request.storage_quota_gb
    )

    # Log action
    AuditLogService.log_action(
        db=db,
        user_id=current_user.id,
        action=AuditAction.UPDATE,
        resource_type="user",
        resource_id=str(user_id),
        description="Updated user quotas",
        changes=request.dict(exclude_none=True)
    )

    return user.to_dict()


@router.post("/users/{user_id}/deactivate", response_model=UserResponse)
@require_admin
async def deactivate_user(
    user_id: int,
    current_user: User = Depends(AuthDependency),
    db: Session = Depends(get_db)
):
    """Deactivate user account (Admin only)."""
    user = UserService.deactivate_user(db, user_id)

    # Log action
    AuditLogService.log_action(
        db=db,
        user_id=current_user.id,
        action=AuditAction.UPDATE,
        resource_type="user",
        resource_id=str(user_id),
        description="Deactivated user account"
    )

    return user.to_dict()


@router.post("/users/{user_id}/activate", response_model=UserResponse)
@require_admin
async def activate_user(
    user_id: int,
    current_user: User = Depends(AuthDependency),
    db: Session = Depends(get_db)
):
    """Activate user account (Admin only)."""
    user = UserService.activate_user(db, user_id)

    # Log action
    AuditLogService.log_action(
        db=db,
        user_id=current_user.id,
        action=AuditAction.UPDATE,
        resource_type="user",
        resource_id=str(user_id),
        description="Activated user account"
    )

    return user.to_dict()


# ============================================================================
# API Key Endpoints
# ============================================================================

@router.post("/api-keys", response_model=APIKeyCreateResponse, status_code=status.HTTP_201_CREATED)
async def create_api_key(
    request: APIKeyCreateRequest,
    current_user: User = Depends(AuthDependency),
    db: Session = Depends(get_db)
):
    """Create a new API key for the current user."""
    api_key, full_key = APIKeyService.create_api_key(
        db=db,
        user_id=current_user.id,
        key_name=request.key_name,
        scopes=request.scopes,
        expires_in_days=request.expires_in_days,
        rate_limit_per_minute=request.rate_limit_per_minute,
        rate_limit_per_hour=request.rate_limit_per_hour
    )

    # Log action
    AuditLogService.log_action(
        db=db,
        user_id=current_user.id,
        action=AuditAction.CREATE,
        resource_type="api_key",
        resource_id=str(api_key.id),
        description=f"Created API key '{request.key_name}'"
    )

    return {
        "api_key": api_key.to_dict(),
        "full_key": full_key,
        "warning": "Store this key securely. It will not be shown again."
    }


@router.get("/api-keys", response_model=List[APIKeyResponse])
async def list_api_keys(
    current_user: User = Depends(AuthDependency),
    include_revoked: bool = False,
    db: Session = Depends(get_db)
):
    """List all API keys for the current user."""
    api_keys = APIKeyService.list_user_api_keys(
        db,
        current_user.id,
        include_revoked
    )

    return [key.to_dict() for key in api_keys]


@router.post("/api-keys/{api_key_id}/revoke", response_model=APIKeyResponse)
async def revoke_api_key(
    api_key_id: int,
    current_user: User = Depends(AuthDependency),
    db: Session = Depends(get_db)
):
    """Revoke an API key."""
    api_key = APIKeyService.revoke_api_key(db, api_key_id, current_user.id)

    # Log action
    AuditLogService.log_action(
        db=db,
        user_id=current_user.id,
        action=AuditAction.UPDATE,
        resource_type="api_key",
        resource_id=str(api_key_id),
        description=f"Revoked API key '{api_key.key_name}'"
    )

    return api_key.to_dict()


@router.delete("/api-keys/{api_key_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_api_key(
    api_key_id: int,
    current_user: User = Depends(AuthDependency),
    db: Session = Depends(get_db)
):
    """Delete an API key."""
    # Get API key name before deletion for logging
    api_key = db.query(APIKey).filter(
        APIKey.id == api_key_id,
        APIKey.user_id == current_user.id
    ).first()

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found"
        )

    key_name = api_key.key_name

    # Delete
    APIKeyService.delete_api_key(db, api_key_id, current_user.id)

    # Log action
    AuditLogService.log_action(
        db=db,
        user_id=current_user.id,
        action=AuditAction.DELETE,
        resource_type="api_key",
        resource_id=str(api_key_id),
        description=f"Deleted API key '{key_name}'"
    )

    return None


# ============================================================================
# Audit Log Endpoints
# ============================================================================

@router.get("/audit-logs/me", response_model=List[AuditLogResponse])
async def get_my_audit_logs(
    current_user: User = Depends(AuthDependency),
    skip: int = 0,
    limit: int = 100,
    action: Optional[AuditAction] = None,
    resource_type: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Get audit logs for the current user.

    Query parameters:
    - skip: Number of records to skip
    - limit: Maximum number of records to return
    - action: Filter by action type
    - resource_type: Filter by resource type
    """
    logs = AuditLogService.get_user_audit_logs(
        db,
        current_user.id,
        skip,
        limit,
        action,
        resource_type
    )

    return [log.to_dict() for log in logs]


@router.get("/audit-logs", response_model=List[AuditLogResponse])
@require_admin
async def get_all_audit_logs(
    current_user: User = Depends(AuthDependency),
    skip: int = 0,
    limit: int = 100,
    action: Optional[AuditAction] = None,
    resource_type: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    db: Session = Depends(get_db)
):
    """
    Get all audit logs (Admin only).

    Query parameters:
    - skip: Number of records to skip
    - limit: Maximum number of records to return
    - action: Filter by action type
    - resource_type: Filter by resource type
    - start_date: Filter by start date
    - end_date: Filter by end date
    """
    logs = AuditLogService.get_all_audit_logs(
        db,
        skip,
        limit,
        action,
        resource_type,
        start_date,
        end_date
    )

    return [log.to_dict() for log in logs]


@router.get("/audit-logs/users/{user_id}", response_model=List[AuditLogResponse])
@require_admin
async def get_user_audit_logs(
    user_id: int,
    current_user: User = Depends(AuthDependency),
    skip: int = 0,
    limit: int = 100,
    action: Optional[AuditAction] = None,
    resource_type: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Get audit logs for a specific user (Admin only).

    Query parameters:
    - skip: Number of records to skip
    - limit: Maximum number of records to return
    - action: Filter by action type
    - resource_type: Filter by resource type
    """
    logs = AuditLogService.get_user_audit_logs(
        db,
        user_id,
        skip,
        limit,
        action,
        resource_type
    )

    return [log.to_dict() for log in logs]


# ============================================================================
# Quota Check Endpoint
# ============================================================================

@router.get("/quota/check")
async def check_quota(
    current_user: User = Depends(AuthDependency),
    db: Session = Depends(get_db)
):
    """Check if user has remaining API quota."""
    has_quota = UserService.check_api_quota(db, current_user.id)

    return {
        "has_quota": has_quota,
        "quota": current_user.api_calls_quota,
        "used": current_user.api_calls_used,
        "remaining": current_user.api_quota_remaining,
        "percentage": current_user.api_quota_percentage
    }


# Import APIKey model
from ocular.database.models import APIKey
