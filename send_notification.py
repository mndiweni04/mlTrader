# send_notification.py
"""
Handles sending email alerts using Gmail.
Reads credentials from GitHub Actions Secrets.
"""

import smtplib
from email.message import EmailMessage
import os # <-- Make sure os is imported

# --- 🔔 CREDENTIALS ARE NOW READ FROM GITHUB SECRETS 🔔 ---
SENDER_EMAIL = os.environ.get("Sender_Email")
SENDER_APP_PASSWORD = os.environ.get("Sender_App_Password")
RECEIVER_EMAIL = os.environ.get("Receiver_Email")
# ---------------------------------------------------------

def send_email(subject, body):
    """
    Sends an email using the credentials from environment variables.
    """
    
    if not SENDER_EMAIL or not SENDER_APP_PASSWORD or not RECEIVER_EMAIL:
        print("--- EMAIL FAILED: One or more environment variables are missing ---")
        print("   (SENDER_EMAIL, SENDER_APP_PASSWORD, RECEIVER_EMAIL)")
        return

    try:
        msg = EmailMessage()
        msg.set_content(body)
        msg['Subject'] = subject
        msg['From'] = SENDER_EMAIL
        msg['To'] = RECEIVER_EMAIL

        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.login(SENDER_EMAIL, SENDER_APP_PASSWORD)
        server.send_message(msg)
        server.quit()
        print("Notification email sent successfully.")
    except smtplib.SMTPAuthenticationError:
        print("--- EMAIL FAILED: SMTP Authentication Error ---")
        print("   1. Is the SENDER_APP_PASSWORD secret correct?")
        print("   2. Is the SENDER_EMAIL secret correct?")
    except Exception as e:
        print(f"--- EMAIL FAILED: An unexpected error occurred: {e} ---")

if __name__ == "__main__":
    # You can no longer test this by running `python send_notification.py`
    # because the secrets only exist inside GitHub Actions.
    print("This script is now designed to be run by the GitHub Action.")