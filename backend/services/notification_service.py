import asyncio
import logging
from typing import Dict, List, Optional

import httpx
import resend
from config import get_settings
from telegram import Bot
from telegram.error import TelegramError

logger = logging.getLogger(__name__)
settings = get_settings()


class NotificationService:
    """Service for sending notifications via email and Telegram"""

    def __init__(self):
        self.resend_client = resend
        resend.api_key = settings.resend_api_key
        self.telegram_bot = (
            Bot(token=settings.telegram_bot_token)
            if settings.telegram_bot_token != "your-telegram-bot-token"
            else None
        )

    async def send_email_notification(
        self,
        to_emails: List[str],
        subject: str,
        html_content: str,
        text_content: Optional[str] = None,
    ) -> bool:
        """Send email notification using Resend"""
        try:
            if settings.resend_api_key == "your-resend-api-key":
                logger.warning(
                    "Resend API key not configured, skipping email notification"
                )
                return False

            for email in to_emails:
                params = {
                    "from": settings.notification_from_email,
                    "to": [email],
                    "subject": subject,
                    "html": html_content,
                }

                if text_content:
                    params["text"] = text_content

                response = resend.Emails.send(params)
                logger.info(
                    f"Email sent successfully to {email}, ID: {response.get('id')}"
                )

            return True
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
            return False

    async def send_telegram_notification(
        self, message: str, chat_id: Optional[str] = None, parse_mode: str = "HTML"
    ) -> bool:
        """Send Telegram notification"""
        try:
            if not self.telegram_bot:
                logger.warning(
                    "Telegram bot not configured, skipping Telegram notification"
                )
                return False

            target_chat_id = chat_id or settings.telegram_chat_id
            if target_chat_id == "your-telegram-chat-id":
                logger.warning(
                    "Telegram chat ID not configured, skipping Telegram notification"
                )
                return False

            await self.telegram_bot.send_message(
                chat_id=target_chat_id, text=message, parse_mode=parse_mode
            )
            logger.info(f"Telegram message sent successfully to chat {target_chat_id}")
            return True
        except TelegramError as e:
            logger.error(f"Failed to send Telegram notification: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending Telegram notification: {e}")
            return False

    async def send_alert_notification(
        self,
        alert_data: Dict,
        notification_channels: List[str],
        user_emails: List[str] = None,
        telegram_chat_id: Optional[str] = None,
    ) -> Dict[str, bool]:
        """Send alert notification through specified channels"""
        results = {}

        # Prepare message content
        severity_emoji = {"critical": "🚨", "high": "⚠️", "medium": "⚡", "low": "ℹ️"}

        emoji = severity_emoji.get(alert_data.get("severity", "medium"), "⚡")

        # Email content
        email_subject = f"{emoji} ATM Alert: {alert_data.get('title', 'Unknown Alert')}"
        email_html = self._generate_email_html(alert_data)
        email_text = self._generate_email_text(alert_data)

        # Telegram content
        telegram_message = self._generate_telegram_message(alert_data, emoji)

        # Send notifications
        tasks = []

        if "email" in notification_channels and user_emails:
            tasks.append(
                self._send_email_task(
                    user_emails, email_subject, email_html, email_text, "email"
                )
            )

        if "telegram" in notification_channels:
            tasks.append(
                self._send_telegram_task(telegram_message, telegram_chat_id, "telegram")
            )

        if tasks:
            task_results = await asyncio.gather(*tasks, return_exceptions=True)
            for channel, result in task_results:
                results[channel] = (
                    result if not isinstance(result, Exception) else False
                )

        return results

    async def _send_email_task(self, emails, subject, html, text, channel):
        """Wrapper task for email sending"""
        result = await self.send_email_notification(emails, subject, html, text)
        return (channel, result)

    async def _send_telegram_task(self, message, chat_id, channel):
        """Wrapper task for Telegram sending"""
        result = await self.send_telegram_notification(message, chat_id)
        return (channel, result)

    def _generate_email_html(self, alert_data: Dict) -> str:
        """Generate HTML email content for alert"""
        severity_colors = {
            "critical": "#dc2626",
            "high": "#ea580c",
            "medium": "#d97706",
            "low": "#059669",
        }

        color = severity_colors.get(alert_data.get("severity", "medium"), "#d97706")

        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>ATM Alert Notification</title>
        </head>
        <body style="margin: 0; padding: 20px; font-family: Arial, sans-serif; background-color: #f5f5f5;">
            <div style="max-width: 600px; margin: 0 auto; background-color: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                <div style="background-color: {color}; color: white; padding: 20px; text-align: center;">
                    <h1 style="margin: 0; font-size: 24px;">ATM Alert Notification</h1>
                </div>
                <div style="padding: 30px;">
                    <h2 style="color: {color}; margin-top: 0;">{alert_data.get('title', 'Alert')}</h2>
                    <p style="font-size: 16px; line-height: 1.5; color: #333;">{alert_data.get('message', 'No details available')}</p>
                    
                    <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 20px 0;">
                        <p style="margin: 5px 0;"><strong>ATM ID:</strong> {alert_data.get('atm_id', 'Unknown')}</p>
                        <p style="margin: 5px 0;"><strong>Severity:</strong> <span style="color: {color}; text-transform: uppercase; font-weight: bold;">{alert_data.get('severity', 'Unknown')}</span></p>
                        <p style="margin: 5px 0;"><strong>Time:</strong> {alert_data.get('triggered_at', 'Unknown')}</p>
                    </div>
                    
                    <div style="text-align: center; margin-top: 30px;">
                        <a href="#" style="background-color: {color}; color: white; padding: 12px 24px; text-decoration: none; border-radius: 5px; display: inline-block;">View Dashboard</a>
                    </div>
                </div>
                <div style="background-color: #f8f9fa; padding: 15px; text-align: center; border-top: 1px solid #e9ecef;">
                    <p style="margin: 0; color: #6c757d; font-size: 14px;">ATM Insights Monitoring System</p>
                </div>
            </div>
        </body>
        </html>
        """

    def _generate_email_text(self, alert_data: Dict) -> str:
        """Generate plain text email content for alert"""
        return f"""
ATM ALERT NOTIFICATION

{alert_data.get('title', 'Alert')}

Details: {alert_data.get('message', 'No details available')}

ATM ID: {alert_data.get('atm_id', 'Unknown')}
Severity: {alert_data.get('severity', 'Unknown').upper()}
Time: {alert_data.get('triggered_at', 'Unknown')}

Please check your ATM Insights dashboard for more information.

---
ATM Insights Monitoring System
        """

    def _generate_telegram_message(self, alert_data: Dict, emoji: str) -> str:
        """Generate Telegram message for alert"""
        return f"""
{emoji} <b>ATM Alert</b>

<b>{alert_data.get('title', 'Alert')}</b>

{alert_data.get('message', 'No details available')}

<b>ATM ID:</b> {alert_data.get('atm_id', 'Unknown')}
<b>Severity:</b> {alert_data.get('severity', 'Unknown').upper()}
<b>Time:</b> {alert_data.get('triggered_at', 'Unknown')}

Please check your dashboard for more details.
        """


# Global instance
notification_service = NotificationService()
