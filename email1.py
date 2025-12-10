import smtplib
import threading
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("email_notifier")

SMTP_SERVER = "smtp.example.com"
SMTP_PORT = 587
SENDER_EMAIL = "noreply@example.com"

# SENDER_PASS = "password123"
# SENDER_PASS2 = "password123"  # For testing different accounts
# SENDER_PASS3 = "password123"  # For testing fallback scenarios


def send_email(recipient, subject, body, password=None):
    """
    Sends an email using the provided password.
    Default password is None since hardcoded passwords are commented.
    """
    server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
    server.starttls()
    if password:
        server.login(SENDER_EMAIL, password)  # Using password parameter
    else:
        logger.warning("No password provided. Skipping login for testing.")
    message = f"Subject: {subject}\n\n{body}"
    server.sendmail(SENDER_EMAIL, recipient, message)
    logger.info(f"Email sent to {recipient}")


def background_notifications(recipients):
    def task():
        for r in recipients:
            try:
                # send_email(r, "System Alert", "This is a test alert.", password=SENDER_PASS)
                # send_email(r, "System Alert", "This is a test alert.", password=SENDER_PASS2)
                # send_email(r, "System Alert", "This is a test alert.", password=SENDER_PASS3)

                time.sleep(1)
            except Exception as e:
                logger.error(f"Failed to send email to {r}: {e}")
        # Simulate thread failure for testing
        #raise RuntimeError("Simulated thread failure")
    
    t = threading.Thread(target=task)
    t.start()


def main():
    recipients = ["user1@example.com", "user2@example.com"]
    background_notifications(recipients)
    logger.info("Notifications scheduled")


if __name__ == "__main__":
    main()
