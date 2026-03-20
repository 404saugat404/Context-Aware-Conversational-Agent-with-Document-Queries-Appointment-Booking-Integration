"""
Email service for sending appointment confirmation emails.

Uses fastapi-mail with Jinja2 HTML templates.
"""

from __future__ import annotations

from pathlib import Path

from fastapi_mail import ConnectionConfig, FastMail, MessageSchema, MessageType

from backend import config
from backend.logger import get_logger

logger = get_logger(__name__)

_mail_configured: bool = bool(config.MAIL_USERNAME and config.MAIL_PASSWORD and config.MAIL_FROM)

if _mail_configured:
    _mail_config = ConnectionConfig(
        MAIL_USERNAME=config.MAIL_USERNAME,
        MAIL_PASSWORD=config.MAIL_PASSWORD,
        MAIL_FROM=config.MAIL_FROM,
        MAIL_PORT=config.MAIL_PORT,
        MAIL_SERVER=config.MAIL_SERVER,
        MAIL_STARTTLS=config.MAIL_STARTTLS,
        MAIL_SSL_TLS=config.MAIL_SSL_TLS,
        USE_CREDENTIALS=config.USE_CREDENTIALS,
        TEMPLATE_FOLDER=Path(config.TEMPLATE_FOLDER),
    )
    _fm = FastMail(_mail_config)
else:
    logger.warning("Email not configured — MAIL_USERNAME/MAIL_PASSWORD/MAIL_FROM missing")
    _fm = None


async def send_appointment_confirmation(
    appointment_id: str,
    name: str,
    email: str,
    phone: str,
    preferred_date: str,
    reason: str | None = None,
) -> bool:
    """
    Send an appointment confirmation email.

    Returns True if sent, False if email is not configured or sending fails.
    """
    if _fm is None:
        logger.warning("Skipping email — mail is not configured")
        return False

    template_body = {
        "appointment_id": appointment_id,
        "name": name,
        "email": email,
        "phone": phone,
        "preferred_date": preferred_date,
        "reason": reason,
    }

    message = MessageSchema(
        subject=f"Appointment Confirmation — {appointment_id}",
        recipients=[email],
        template_body=template_body,
        subtype=MessageType.html,
    )

    try:
        await _fm.send_message(message, template_name="appointment_confirmation.html")
        logger.info("Confirmation email sent to %s for appointment %s", email, appointment_id)
        return True
    except Exception as exc:
        logger.error("Failed to send confirmation email to %s: %s", email, exc)
        return False
