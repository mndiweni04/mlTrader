# send_notification.py
"""
Handles sending email alerts using Gmail.
"""

import smtplib
from email.message import EmailMessage

# --- 🔔 ACTION REQUIRED 🔔 ---
# ENTER YOUR SENDER EMAIL, 16-DIGIT APP PASSWORD, AND RECEIVER EMAIL
SENDER_EMAIL = "your-email@gmail.com"
SENDER_APP_PASSWORD = "your-16-digit-app-password" # e.g., "abcd efgh ijkl mnop"
RECEIVER_EMAIL = "your-email-to-receive-alerts@gmail.com"
# -----------------------------

def send_email(subject, body):
    """
    Sends an email using the credentials defined above.
    """
    
    if SENDER_EMAIL == "your-email@gmail.com" or SENDER_APP_PASSWORD == "your-16-digit-app-password":
        print("--- EMAIL FAILED: Please configure SENDER_EMAIL and SENDER_APP_PASSWORD in send_notification.py ---")
        return

    try:
        msg = EmailMessage()
        msg.set_content(body)
        msg['Subject'] = subject
        msg['From'] = SENDER_EMAIL
        msg['To'] = RECEIVER_EMAIL

        # Connect to Gmail's SMTP server
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.login(SENDER_EMAIL, SENDER_APP_PASSWORD)
        server.send_message(msg)
        server.quit()
        print("Notification email sent successfully.")
    except smtplib.SMTPAuthenticationError:
        print("--- EMAIL FAILED: SMTP Authentication Error ---")
        print("   1. Did you use the 16-digit 'App Password'?")
        print("   2. Is the SENDER_EMAIL correct?")
    except Exception as e:
        print(f"--- EMAIL FAILED: An unexpected error occurred: {e} ---")

if __name__ == "__main__":
    # This allows you to test the email script directly
    # Run: python send_notification.py
    print("Sending test email...")
    send_email(
        subject="ML Trader Bot - Test Email",
        body="This is a test of the notification system."
    )