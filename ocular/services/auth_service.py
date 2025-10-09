"""
Authentication Service with Clerk.com Integration.

Provides user authentication, authorization, and session management.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import hashlib
import secrets
import httpx
from sqlalchemy.orm import Session
from fastapi import HTTPException, status

from ocular.database.models import (
    User, APIKey, AuditLog, UserRole, APIKeyStatus, AuditAction
)


class ClerkAuthService:
    """
    Clerk.com authentication integration service.

    Handles user authentication and synchronization with Clerk.com.
    """

    def __init__(self, clerk_secret_key: str, clerk_publishable_key: str):
        """
        Initialize Clerk auth service.

        Args:
            clerk_secret_key: Clerk secret API key
            clerk_publishable_key: Clerk publishable key
        """
        self.clerk_secret_key = clerk_secret_key
        self.clerk_publishable_key = clerk_publishable_key
        self.base_url = "https://api.clerk.com/v1"

    async def verify_session(self, session_token: str) -> Dict[str, Any]:
        """
        Verify a Clerk session token.

        Args:
            session_token: Clerk session token

        Returns:
            Session data from Clerk

        Raises:
            HTTPException: If session is invalid
        """
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/sessions/{session_token}/verify",
                headers={"Authorization": f"Bearer {self.clerk_secret_key}"}
            )

            if response.status_code != 200:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid session token"
                )

            return response.json()

    async def get_user(self, clerk_user_id: str) -> Dict[str, Any]:
        """
        Get user data from Clerk.

        Args:
            clerk_user_id: Clerk user ID

        Returns:
            User data from Clerk
        """
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/users/{clerk_user_id}",
                headers={"Authorization": f"Bearer {self.clerk_secret_key}"}
            )

            if response.status_code != 200:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="User not found in Clerk"
                )

            return response.json()

    async def sync_user_from_clerk(
        self,
        db: Session,
        clerk_user_id: str
    ) -> User:
        """
        Sync user data from Clerk to local database.

        Args:
            db: Database session
            clerk_user_id: Clerk user ID

        Returns:
            Local user object
        """
        # Get user data from Clerk
        clerk_data = await self.get_user(clerk_user_id)

        # Check if user exists locally
        user = db.query(User).filter(User.clerk_user_id == clerk_user_id).first()

        # Extract email from Clerk data
        email_addresses = clerk_data.get("email_addresses", [])
        primary_email = None
        for email_obj in email_addresses:
            if email_obj.get("id") == clerk_data.get("primary_email_address_id"):
                primary_email = email_obj.get("email_address")
                break

        if not primary_email and email_addresses:
            primary_email = email_addresses[0].get("email_address")

        if not primary_email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No email address found for user"
            )

        if user:
            # Update existing user
            user.email = primary_email
            user.username = clerk_data.get("username")
            user.first_name = clerk_data.get("first_name")
            user.last_name = clerk_data.get("last_name")
            user.avatar_url = clerk_data.get("image_url")
            user.is_verified = clerk_data.get("email_addresses", [{}])[0].get("verification", {}).get("status") == "verified"
            user.updated_at = datetime.utcnow()
        else:
            # Create new user
            user = User(
                clerk_user_id=clerk_user_id,
                email=primary_email,
                username=clerk_data.get("username"),
                first_name=clerk_data.get("first_name"),
                last_name=clerk_data.get("last_name"),
                avatar_url=clerk_data.get("image_url"),
                is_verified=clerk_data.get("email_addresses", [{}])[0].get("verification", {}).get("status") == "verified",
                role=UserRole.USER,
                is_active=True
            )
            db.add(user)

        db.commit()
        db.refresh(user)

        return user


class UserService:
    """Service for user management operations."""

    @staticmethod
    def get_user_by_id(db: Session, user_id: int) -> Optional[User]:
        """Get user by ID."""
        return db.query(User).filter(User.id == user_id).first()

    @staticmethod
    def get_user_by_clerk_id(db: Session, clerk_user_id: str) -> Optional[User]:
        """Get user by Clerk user ID."""
        return db.query(User).filter(User.clerk_user_id == clerk_user_id).first()

    @staticmethod
    def get_user_by_email(db: Session, email: str) -> Optional[User]:
        """Get user by email."""
        return db.query(User).filter(User.email == email).first()

    @staticmethod
    def list_users(
        db: Session,
        skip: int = 0,
        limit: int = 100,
        role: Optional[UserRole] = None,
        is_active: Optional[bool] = None
    ) -> List[User]:
        """
        List users with optional filters.

        Args:
            db: Database session
            skip: Number of records to skip
            limit: Maximum number of records to return
            role: Filter by role
            is_active: Filter by active status

        Returns:
            List of users
        """
        query = db.query(User)

        if role:
            query = query.filter(User.role == role)
        if is_active is not None:
            query = query.filter(User.is_active == is_active)

        return query.offset(skip).limit(limit).all()

    @staticmethod
    def update_user_role(
        db: Session,
        user_id: int,
        new_role: UserRole
    ) -> User:
        """
        Update user's role.

        Args:
            db: Database session
            user_id: User ID
            new_role: New role to assign

        Returns:
            Updated user
        """
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        user.role = new_role
        user.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(user)

        return user

    @staticmethod
    def update_user_quotas(
        db: Session,
        user_id: int,
        api_calls_quota: Optional[int] = None,
        deployment_quota: Optional[int] = None,
        storage_quota_gb: Optional[float] = None
    ) -> User:
        """
        Update user's quotas.

        Args:
            db: Database session
            user_id: User ID
            api_calls_quota: New API calls quota
            deployment_quota: New deployment quota
            storage_quota_gb: New storage quota in GB

        Returns:
            Updated user
        """
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        if api_calls_quota is not None:
            user.api_calls_quota = api_calls_quota
        if deployment_quota is not None:
            user.deployment_quota = deployment_quota
        if storage_quota_gb is not None:
            user.storage_quota_gb = storage_quota_gb

        user.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(user)

        return user

    @staticmethod
    def increment_api_usage(db: Session, user_id: int) -> User:
        """
        Increment user's API usage counter.

        Args:
            db: Database session
            user_id: User ID

        Returns:
            Updated user
        """
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        user.api_calls_used += 1
        db.commit()
        db.refresh(user)

        return user

    @staticmethod
    def check_api_quota(db: Session, user_id: int) -> bool:
        """
        Check if user has remaining API quota.

        Args:
            db: Database session
            user_id: User ID

        Returns:
            True if user has remaining quota, False otherwise
        """
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            return False

        return user.api_calls_used < user.api_calls_quota

    @staticmethod
    def reset_monthly_quota(db: Session, user_id: int) -> User:
        """
        Reset user's monthly API quota.

        Args:
            db: Database session
            user_id: User ID

        Returns:
            Updated user
        """
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        user.api_calls_used = 0
        user.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(user)

        return user

    @staticmethod
    def deactivate_user(db: Session, user_id: int) -> User:
        """
        Deactivate a user account.

        Args:
            db: Database session
            user_id: User ID

        Returns:
            Updated user
        """
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        user.is_active = False
        user.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(user)

        return user

    @staticmethod
    def activate_user(db: Session, user_id: int) -> User:
        """
        Activate a user account.

        Args:
            db: Database session
            user_id: User ID

        Returns:
            Updated user
        """
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        user.is_active = True
        user.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(user)

        return user


class APIKeyService:
    """Service for API key management."""

    @staticmethod
    def generate_api_key() -> tuple[str, str, str]:
        """
        Generate a new API key.

        Returns:
            Tuple of (full_key, prefix, hash)
        """
        # Generate random API key (32 bytes = 64 hex characters)
        full_key = f"ocular_{secrets.token_urlsafe(32)}"

        # Extract prefix (first 12 characters after 'ocular_')
        prefix = full_key[:19]  # 'ocular_' + 12 chars

        # Hash the full key for storage
        key_hash = hashlib.sha256(full_key.encode()).hexdigest()

        return full_key, prefix, key_hash

    @staticmethod
    def create_api_key(
        db: Session,
        user_id: int,
        key_name: str,
        scopes: Optional[List[str]] = None,
        expires_in_days: Optional[int] = None,
        rate_limit_per_minute: Optional[int] = None,
        rate_limit_per_hour: Optional[int] = None
    ) -> tuple[APIKey, str]:
        """
        Create a new API key for a user.

        Args:
            db: Database session
            user_id: User ID
            key_name: User-defined name for the key
            scopes: List of allowed operations
            expires_in_days: Number of days until expiration
            rate_limit_per_minute: Rate limit per minute
            rate_limit_per_hour: Rate limit per hour

        Returns:
            Tuple of (APIKey object, full_key_string)
        """
        # Generate key
        full_key, prefix, key_hash = APIKeyService.generate_api_key()

        # Calculate expiration
        expires_at = None
        if expires_in_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_in_days)

        # Create API key
        api_key = APIKey(
            user_id=user_id,
            key_name=key_name,
            key_prefix=prefix,
            key_hash=key_hash,
            scopes=scopes or [],
            status=APIKeyStatus.ACTIVE,
            expires_at=expires_at,
            rate_limit_per_minute=rate_limit_per_minute,
            rate_limit_per_hour=rate_limit_per_hour
        )

        db.add(api_key)
        db.commit()
        db.refresh(api_key)

        return api_key, full_key

    @staticmethod
    def verify_api_key(db: Session, api_key_string: str) -> Optional[APIKey]:
        """
        Verify an API key and return the associated key object.

        Args:
            db: Database session
            api_key_string: Full API key string

        Returns:
            APIKey object if valid, None otherwise
        """
        # Hash the provided key
        key_hash = hashlib.sha256(api_key_string.encode()).hexdigest()

        # Find the key
        api_key = db.query(APIKey).filter(APIKey.key_hash == key_hash).first()

        if not api_key:
            return None

        # Check if key is valid
        if not api_key.is_valid:
            return None

        # Update last used timestamp
        api_key.last_used_at = datetime.utcnow()
        api_key.usage_count += 1
        db.commit()

        return api_key

    @staticmethod
    def list_user_api_keys(
        db: Session,
        user_id: int,
        include_revoked: bool = False
    ) -> List[APIKey]:
        """
        List all API keys for a user.

        Args:
            db: Database session
            user_id: User ID
            include_revoked: Include revoked keys in results

        Returns:
            List of API keys
        """
        query = db.query(APIKey).filter(APIKey.user_id == user_id)

        if not include_revoked:
            query = query.filter(APIKey.status == APIKeyStatus.ACTIVE)

        return query.all()

    @staticmethod
    def revoke_api_key(db: Session, api_key_id: int, user_id: int) -> APIKey:
        """
        Revoke an API key.

        Args:
            db: Database session
            api_key_id: API key ID
            user_id: User ID (for authorization)

        Returns:
            Updated API key
        """
        api_key = db.query(APIKey).filter(
            APIKey.id == api_key_id,
            APIKey.user_id == user_id
        ).first()

        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="API key not found"
            )

        api_key.status = APIKeyStatus.REVOKED
        api_key.revoked_at = datetime.utcnow()
        api_key.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(api_key)

        return api_key

    @staticmethod
    def delete_api_key(db: Session, api_key_id: int, user_id: int) -> bool:
        """
        Delete an API key.

        Args:
            db: Database session
            api_key_id: API key ID
            user_id: User ID (for authorization)

        Returns:
            True if deleted successfully
        """
        api_key = db.query(APIKey).filter(
            APIKey.id == api_key_id,
            APIKey.user_id == user_id
        ).first()

        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="API key not found"
            )

        db.delete(api_key)
        db.commit()

        return True


class AuditLogService:
    """Service for audit logging."""

    @staticmethod
    def log_action(
        db: Session,
        user_id: Optional[int],
        action: AuditAction,
        resource_type: str,
        resource_id: Optional[str] = None,
        description: Optional[str] = None,
        changes: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        request_method: Optional[str] = None,
        request_path: Optional[str] = None,
        status: str = "success",
        error_message: Optional[str] = None
    ) -> AuditLog:
        """
        Log an action to the audit log.

        Args:
            db: Database session
            user_id: User ID (None for system actions)
            action: Action type
            resource_type: Type of resource affected
            resource_id: ID of affected resource
            description: Human-readable description
            changes: Before/after changes
            metadata: Additional metadata
            ip_address: Client IP address
            user_agent: Client user agent
            request_method: HTTP method
            request_path: Request path
            status: Action status (success, failure, error)
            error_message: Error message if failed

        Returns:
            Created audit log entry
        """
        audit_log = AuditLog(
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            description=description,
            changes=changes,
            log_metadata=metadata,
            ip_address=ip_address,
            user_agent=user_agent,
            request_method=request_method,
            request_path=request_path,
            status=status,
            error_message=error_message
        )

        db.add(audit_log)
        db.commit()
        db.refresh(audit_log)

        return audit_log

    @staticmethod
    def get_user_audit_logs(
        db: Session,
        user_id: int,
        skip: int = 0,
        limit: int = 100,
        action: Optional[AuditAction] = None,
        resource_type: Optional[str] = None
    ) -> List[AuditLog]:
        """
        Get audit logs for a specific user.

        Args:
            db: Database session
            user_id: User ID
            skip: Number of records to skip
            limit: Maximum number of records to return
            action: Filter by action type
            resource_type: Filter by resource type

        Returns:
            List of audit logs
        """
        query = db.query(AuditLog).filter(AuditLog.user_id == user_id)

        if action:
            query = query.filter(AuditLog.action == action)
        if resource_type:
            query = query.filter(AuditLog.resource_type == resource_type)

        return query.order_by(AuditLog.created_at.desc()).offset(skip).limit(limit).all()

    @staticmethod
    def get_all_audit_logs(
        db: Session,
        skip: int = 0,
        limit: int = 100,
        action: Optional[AuditAction] = None,
        resource_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[AuditLog]:
        """
        Get all audit logs with optional filters.

        Args:
            db: Database session
            skip: Number of records to skip
            limit: Maximum number of records to return
            action: Filter by action type
            resource_type: Filter by resource type
            start_date: Filter by start date
            end_date: Filter by end date

        Returns:
            List of audit logs
        """
        query = db.query(AuditLog)

        if action:
            query = query.filter(AuditLog.action == action)
        if resource_type:
            query = query.filter(AuditLog.resource_type == resource_type)
        if start_date:
            query = query.filter(AuditLog.created_at >= start_date)
        if end_date:
            query = query.filter(AuditLog.created_at <= end_date)

        return query.order_by(AuditLog.created_at.desc()).offset(skip).limit(limit).all()
