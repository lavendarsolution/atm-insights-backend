#!/usr/bin/env python3
"""
Test script for notification services
"""
import asyncio
import logging
import os
import sys

# Add the backend directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

from config import get_settings
from services.notification_service import NotificationService

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def test_notifications():
    """Test both email and Telegram notifications"""

    settings = get_settings()
    notification_service = NotificationService()

    print("🧪 Testing ATM Insights Notification System")
    print("=" * 50)

    # Test data
    test_alert_data = {
        "title": "Test Alert - System Check",
        "message": "This is a test alert to verify notification system is working correctly.",
        "atm_id": "TEST-001",
        "severity": "medium",
        "triggered_at": "2025-06-03T12:00:00Z",
        "rule_name": "Test Rule",
    }

    print(f"📧 Email Configuration:")
    print(f"   From: {settings.notification_from_email}")
    print(f"   Target: {settings.notification_target_email}")
    print(
        f"   API Key configured: {bool(settings.resend_api_key and settings.resend_api_key != 'your-resend-api-key')}"
    )

    print(f"\n📱 Telegram Configuration:")
    print(
        f"   Bot Token: {settings.telegram_bot_token[:10]}... (length: {len(settings.telegram_bot_token)})"
    )
    print(f"   Chat ID: {settings.telegram_chat_id}")
    print(f"   Bot initialized: {bool(notification_service.telegram_bot)}")

    print(f"\n🚀 Sending test notifications...")

    # Test notifications
    results = await notification_service.send_alert_notification(
        alert_data=test_alert_data,
        notification_channels=["email", "telegram"],
        user_emails=[settings.notification_target_email],
    )

    print(f"\n📊 Results:")
    for channel, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"   {channel.upper()}: {status}")

    # Test individual services
    print(f"\n🔍 Individual Service Tests:")

    # Test Telegram
    try:
        telegram_result = await notification_service.send_telegram_notification(
            message="🧪 <b>Test Message</b>\n\nThis is a direct Telegram test from the notification service.",
            chat_id=settings.telegram_chat_id,
        )
        print(f"   Direct Telegram: {'✅ SUCCESS' if telegram_result else '❌ FAILED'}")
    except Exception as e:
        print(f"   Direct Telegram: ❌ FAILED - {e}")

    # Test Email
    try:
        email_result = await notification_service.send_email_notification(
            to_emails=[settings.notification_target_email],
            subject="🧪 Test Email - ATM Insights",
            html_content="<h2>Test Email</h2><p>This is a direct email test from the notification service.</p>",
            text_content="Test Email\n\nThis is a direct email test from the notification service.",
        )
        print(f"   Direct Email: {'✅ SUCCESS' if email_result else '❌ FAILED'}")
    except Exception as e:
        print(f"   Direct Email: ❌ FAILED - {e}")


if __name__ == "__main__":
    print("Starting notification test...")
    try:
        asyncio.run(test_notifications())
        print("\n✅ Test completed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        logger.exception("Test failed with exception")
