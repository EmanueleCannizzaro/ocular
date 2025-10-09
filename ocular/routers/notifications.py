"""
Notifications API Router - Real-time notification system.
"""
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import or_
from datetime import datetime

from ocular.database.session import get_db
from ocular.database.models import Notification


router = APIRouter()


class NotificationResponse(BaseModel):
    """Notification response model."""
    id: int
    type: str
    message: str
    link: Optional[str]
    read: bool
    created_at: datetime

    class Config:
        from_attributes = True


class NotificationCreate(BaseModel):
    """Notification creation model."""
    type: str
    message: str
    link: Optional[str] = None
    user_id: Optional[int] = None


@router.get("/notifications/unread", response_model=List[NotificationResponse])
async def get_unread_notifications(
    user_id: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """Get all unread notifications for a user."""
    try:
        query = db.query(Notification).filter(Notification.read == False)

        if user_id:
            query = query.filter(
                or_(
                    Notification.user_id == user_id,
                    Notification.user_id == None  # Global notifications
                )
            )
        else:
            query = query.filter(Notification.user_id == None)

        notifications = query.order_by(Notification.created_at.desc()).limit(10).all()
        return notifications

    except Exception as e:
        print(f"Error loading notifications: {e}")
        # Return empty list if table doesn't exist yet
        return []


@router.get("/notifications", response_model=List[NotificationResponse])
async def get_all_notifications(
    user_id: Optional[int] = None,
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """Get all notifications for a user with pagination."""
    try:
        query = db.query(Notification)

        if user_id:
            query = query.filter(
                or_(
                    Notification.user_id == user_id,
                    Notification.user_id == None
                )
            )
        else:
            query = query.filter(Notification.user_id == None)

        notifications = query.order_by(
            Notification.created_at.desc()
        ).limit(limit).offset(offset).all()

        return notifications

    except Exception as e:
        print(f"Error loading notifications: {e}")
        return []


@router.post("/notifications/{notification_id}/read")
async def mark_notification_as_read(
    notification_id: int,
    db: Session = Depends(get_db)
):
    """Mark a specific notification as read."""
    try:
        notification = db.query(Notification).filter(
            Notification.id == notification_id
        ).first()

        if not notification:
            raise HTTPException(status_code=404, detail="Notification not found")

        notification.read = True
        notification.read_at = datetime.utcnow()
        db.commit()

        return {"status": "success", "message": "Notification marked as read"}

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to mark notification as read: {str(e)}")


@router.post("/notifications/mark-all-read")
async def mark_all_notifications_as_read(
    user_id: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """Mark all unread notifications as read for a user."""
    try:
        query = db.query(Notification).filter(Notification.read == False)

        if user_id:
            query = query.filter(
                or_(
                    Notification.user_id == user_id,
                    Notification.user_id == None
                )
            )
        else:
            query = query.filter(Notification.user_id == None)

        count = query.update({
            "read": True,
            "read_at": datetime.utcnow()
        })

        db.commit()

        return {"status": "success", "count": count, "message": f"{count} notifications marked as read"}

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to mark all as read: {str(e)}")


@router.post("/notifications", response_model=NotificationResponse)
async def create_notification(
    notification: NotificationCreate,
    db: Session = Depends(get_db)
):
    """Create a new notification (admin/system use)."""
    try:
        db_notification = Notification(
            type=notification.type,
            message=notification.message,
            link=notification.link,
            user_id=notification.user_id,
            read=False,
            created_at=datetime.utcnow()
        )

        db.add(db_notification)
        db.commit()
        db.refresh(db_notification)

        return db_notification

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create notification: {str(e)}")


@router.delete("/notifications/{notification_id}")
async def delete_notification(
    notification_id: int,
    db: Session = Depends(get_db)
):
    """Delete a notification."""
    try:
        notification = db.query(Notification).filter(
            Notification.id == notification_id
        ).first()

        if not notification:
            raise HTTPException(status_code=404, detail="Notification not found")

        db.delete(notification)
        db.commit()

        return {"status": "success", "message": "Notification deleted"}

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to delete notification: {str(e)}")
