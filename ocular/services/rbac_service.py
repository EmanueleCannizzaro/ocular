"""
Role-Based Access Control (RBAC) Service.

Provides permission checking and access control functionality.
"""

from typing import List, Optional, Callable
from functools import wraps
from fastapi import HTTPException, status, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from ocular.database.models import User, UserRole, APIKey
from ocular.database.session import get_db
from ocular.services.auth_service import ClerkAuthService, APIKeyService


# Permission definitions
class Permissions:
    """Define all available permissions."""

    # Model permissions
    MODEL_READ = "model:read"
    MODEL_CREATE = "model:create"
    MODEL_UPDATE = "model:update"
    MODEL_DELETE = "model:delete"

    # Deployment permissions
    DEPLOYMENT_READ = "deployment:read"
    DEPLOYMENT_CREATE = "deployment:create"
    DEPLOYMENT_UPDATE = "deployment:update"
    DEPLOYMENT_DELETE = "deployment:delete"
    DEPLOYMENT_START = "deployment:start"
    DEPLOYMENT_STOP = "deployment:stop"

    # Provider permissions
    PROVIDER_READ = "provider:read"
    PROVIDER_CREATE = "provider:create"
    PROVIDER_UPDATE = "provider:update"
    PROVIDER_DELETE = "provider:delete"

    # Version permissions
    VERSION_READ = "version:read"
    VERSION_CREATE = "version:create"
    VERSION_UPDATE = "version:update"
    VERSION_DELETE = "version:delete"
    VERSION_ROLLBACK = "version:rollback"

    # API key permissions
    APIKEY_READ = "apikey:read"
    APIKEY_CREATE = "apikey:create"
    APIKEY_REVOKE = "apikey:revoke"
    APIKEY_DELETE = "apikey:delete"

    # User management permissions
    USER_READ = "user:read"
    USER_UPDATE = "user:update"
    USER_DELETE = "user:delete"
    USER_MANAGE_ROLES = "user:manage_roles"
    USER_MANAGE_QUOTAS = "user:manage_quotas"

    # Audit log permissions
    AUDIT_READ = "audit:read"
    AUDIT_EXPORT = "audit:export"

    # Analytics permissions
    ANALYTICS_READ = "analytics:read"
    ANALYTICS_EXPORT = "analytics:export"

    # Admin permissions
    ADMIN_ALL = "admin:*"


# Role-Permission mapping
ROLE_PERMISSIONS = {
    UserRole.VIEWER: [
        Permissions.MODEL_READ,
        Permissions.DEPLOYMENT_READ,
        Permissions.PROVIDER_READ,
        Permissions.VERSION_READ,
        Permissions.ANALYTICS_READ,
    ],
    UserRole.USER: [
        # All viewer permissions
        Permissions.MODEL_READ,
        Permissions.DEPLOYMENT_READ,
        Permissions.PROVIDER_READ,
        Permissions.VERSION_READ,
        Permissions.ANALYTICS_READ,
        # User-specific permissions
        Permissions.MODEL_CREATE,
        Permissions.MODEL_UPDATE,
        Permissions.DEPLOYMENT_CREATE,
        Permissions.DEPLOYMENT_UPDATE,
        Permissions.DEPLOYMENT_START,
        Permissions.DEPLOYMENT_STOP,
        Permissions.VERSION_CREATE,
        Permissions.VERSION_UPDATE,
        Permissions.APIKEY_READ,
        Permissions.APIKEY_CREATE,
        Permissions.APIKEY_REVOKE,
    ],
    UserRole.ADMIN: [
        Permissions.ADMIN_ALL,  # Admin has all permissions
    ]
}


security = HTTPBearer()


class RBACService:
    """Role-Based Access Control service."""

    @staticmethod
    def has_permission(user: User, permission: str) -> bool:
        """
        Check if user has a specific permission.

        Args:
            user: User object
            permission: Permission string

        Returns:
            True if user has permission, False otherwise
        """
        if not user.is_active:
            return False

        # Admin has all permissions
        if user.role == UserRole.ADMIN:
            return True

        # Get permissions for user's role
        role_perms = ROLE_PERMISSIONS.get(user.role, [])

        return permission in role_perms

    @staticmethod
    def check_permission(user: User, permission: str) -> None:
        """
        Check if user has permission, raise exception if not.

        Args:
            user: User object
            permission: Permission string

        Raises:
            HTTPException: If user doesn't have permission
        """
        if not RBACService.has_permission(user, permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission denied: {permission}"
            )

    @staticmethod
    def has_any_permission(user: User, permissions: List[str]) -> bool:
        """
        Check if user has any of the specified permissions.

        Args:
            user: User object
            permissions: List of permission strings

        Returns:
            True if user has any permission, False otherwise
        """
        for permission in permissions:
            if RBACService.has_permission(user, permission):
                return True
        return False

    @staticmethod
    def has_all_permissions(user: User, permissions: List[str]) -> bool:
        """
        Check if user has all of the specified permissions.

        Args:
            user: User object
            permissions: List of permission strings

        Returns:
            True if user has all permissions, False otherwise
        """
        for permission in permissions:
            if not RBACService.has_permission(user, permission):
                return False
        return True

    @staticmethod
    def is_admin(user: User) -> bool:
        """Check if user is an admin."""
        return user.is_active and user.role == UserRole.ADMIN

    @staticmethod
    def is_user_or_admin(user: User) -> bool:
        """Check if user has at least USER role."""
        return user.is_active and user.role in [UserRole.USER, UserRole.ADMIN]

    @staticmethod
    def can_access_resource(user: User, resource_owner_id: int) -> bool:
        """
        Check if user can access a resource.

        Admins can access all resources, users can only access their own.

        Args:
            user: User object
            resource_owner_id: ID of the resource owner

        Returns:
            True if user can access the resource
        """
        if user.role == UserRole.ADMIN:
            return True

        return user.id == resource_owner_id


class AuthDependency:
    """Authentication dependency for FastAPI routes."""

    def __init__(self, clerk_secret_key: str, clerk_publishable_key: str):
        """
        Initialize auth dependency.

        Args:
            clerk_secret_key: Clerk secret API key
            clerk_publishable_key: Clerk publishable key
        """
        self.clerk_service = ClerkAuthService(clerk_secret_key, clerk_publishable_key)

    async def __call__(
        self,
        request: Request,
        credentials: HTTPAuthorizationCredentials = Depends(security),
        db: Session = Depends(get_db)
    ) -> User:
        """
        Authenticate user from request.

        Supports both Clerk session tokens and API keys.

        Args:
            request: FastAPI request object
            credentials: HTTP authorization credentials
            db: Database session

        Returns:
            Authenticated user

        Raises:
            HTTPException: If authentication fails
        """
        token = credentials.credentials

        # Check if it's an API key (starts with 'ocular_')
        if token.startswith('ocular_'):
            return await self._authenticate_with_api_key(db, token, request)
        else:
            return await self._authenticate_with_clerk(db, token)

    async def _authenticate_with_clerk(
        self,
        db: Session,
        session_token: str
    ) -> User:
        """
        Authenticate with Clerk session token.

        Args:
            db: Database session
            session_token: Clerk session token

        Returns:
            Authenticated user
        """
        try:
            # Verify session with Clerk
            session_data = await self.clerk_service.verify_session(session_token)
            clerk_user_id = session_data.get("user_id")

            if not clerk_user_id:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid session token"
                )

            # Sync user from Clerk
            user = await self.clerk_service.sync_user_from_clerk(db, clerk_user_id)

            # Update last login
            user.last_login_at = datetime.utcnow()
            db.commit()

            return user

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Authentication failed: {str(e)}"
            )

    async def _authenticate_with_api_key(
        self,
        db: Session,
        api_key_string: str,
        request: Request
    ) -> User:
        """
        Authenticate with API key.

        Args:
            db: Database session
            api_key_string: API key string
            request: FastAPI request object

        Returns:
            Authenticated user
        """
        # Verify API key
        api_key = APIKeyService.verify_api_key(db, api_key_string)

        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key"
            )

        # Get user
        user = db.query(User).filter(User.id == api_key.user_id).first()

        if not user or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found or inactive"
            )

        # Store API key info in request state for rate limiting
        request.state.api_key = api_key

        return user


def require_permission(permission: str):
    """
    Decorator to require a specific permission for a route.

    Args:
        permission: Permission string required

    Returns:
        Decorator function
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract user from kwargs
            user = kwargs.get('current_user')
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required"
                )

            # Check permission
            RBACService.check_permission(user, permission)

            return await func(*args, **kwargs)

        return wrapper
    return decorator


def require_role(role: UserRole):
    """
    Decorator to require a specific role for a route.

    Args:
        role: Required user role

    Returns:
        Decorator function
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract user from kwargs
            user = kwargs.get('current_user')
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required"
                )

            # Check role
            if user.role != role and user.role != UserRole.ADMIN:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Role {role.value} required"
                )

            return await func(*args, **kwargs)

        return wrapper
    return decorator


def require_admin(func: Callable):
    """
    Decorator to require admin role for a route.

    Args:
        func: Function to decorate

    Returns:
        Decorated function
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Extract user from kwargs
        user = kwargs.get('current_user')
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required"
            )

        # Check admin
        if not RBACService.is_admin(user):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required"
            )

        return await func(*args, **kwargs)

    return wrapper


# Import datetime for last_login_at
from datetime import datetime
