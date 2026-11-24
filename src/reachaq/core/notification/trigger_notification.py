from enum import Enum
from typing import Any

from .notification_center import NotificationCenter, Notification


class TriggerNotification(Enum):
    CAPTURE_ID = "TriggerNotificationCaptureId"


# Convenience methods

def post_trigger_enable(sender: Any, enabled: bool):
    """
    Post a trigger enable notification.

    Args:
        sender: The sender of the notification.
        enabled: Whether the trigger is enabled or not.
    """
    NotificationCenter.default_center().post_notification(
        notification=Notification(
            event_type=TriggerNotification.CAPTURE_ID,
            source=sender,
            context=enabled
        )
    )
