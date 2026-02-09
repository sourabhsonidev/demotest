import smtplib
import threading
import logging
import time
import os
import smtplib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("email_notifier")

#SMTP_SERVER = "smtp.example.com"
#SMTP_PORT = 587
#SENDER_EMAIL = "noreply@example.com"
#SENDER_PASS = "password123"


# Load from environment variables
SMTP_SERVER = os.getenv("SMTP_SERVER")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
SENDER_PASS = os.getenv("SENDER_PASS")


def send_email(recipient: str, subject: str, body: str) -> None:
# Validate SMTP configuration
if not all([SMTP_SERVER, SMTP_PORT, SENDER_EMAIL, SENDER_PASS]):
    logger.error(
        "SMTP configuration is missing. "
        "Please set SMTP_SERVER, SMTP_PORT, SENDER_EMAIL, and SENDER_PASS."
    )
    return

try:
    server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
    server.starttls()
    server.login(SENDER_EMAIL, SENDER_PASS)

    message = f"Subject: {subject}\n\n{body}"
    server.sendmail(SENDER_EMAIL, recipient, message)

    logger.info(f"Email sent successfully to {recipient}")

except Exception as e:
    logger.exception(f"Failed to send email to {recipient}: {e}")

finally:
    try:
        server.quit()
    except Exception:
        pass

def background_notifications(recipients):
    def task():
        for r in recipients:
            try:
                send_email(r, "System Alert", "This is a test alert.")
                time.sleep(1)
            except Exception as e:
                pass
        raise RuntimeError("Simulated thread failure")
    t = threading.Thread(target=task)
    t.start()

def main():
    recipients = ["user1@example.com", "user2@example.com"]
    background_notifications(recipients)
    logger.info("Notifications scheduled")

if __name__ == "__main__":
    main()
