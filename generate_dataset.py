"""
generate_dataset.py
--------------------
Generates a labeled dataset of SMS/WhatsApp-style messages for the
AI-Powered Scam Message Detector project.

Labels: SAFE, SUSPICIOUS, SCAM

This creates realistic, varied text using template + slot-filling so the
dataset has enough lexical diversity for TF-IDF based models to learn
genuine patterns (urgency words, links, money/prize language, etc.)
rather than memorizing a handful of fixed sentences.
"""

import random
import csv

random.seed(42)

# ---------------------------------------------------------------------------
# SCAM messages (clear fraud: phishing, prize scams, fake bank alerts, etc.)
# ---------------------------------------------------------------------------
scam_templates = [
    "Congratulations! You have WON ${amount}! Click {link} now to claim your prize before it expires.",
    "URGENT: Your bank account has been suspended. Verify your details immediately at {link} or lose access.",
    "You've been selected for a free {item}! Claim now: {link}. Limited time offer, act fast!",
    "Dear customer, your package could not be delivered. Pay a customs fee of ${amount} here: {link}",
    "Your account will be permanently closed in 24 hours. Confirm your password at {link} to keep it active.",
    "FINAL NOTICE: You owe ${amount} in unpaid taxes. Pay now at {link} to avoid arrest.",
    "You have an unclaimed inheritance of ${amount}. Contact us immediately at {link} to process the transfer.",
    "Your card ending in {digits} was charged ${amount}. If this wasn't you, click {link} to dispute now.",
    "Hi, this is {bank} security. Suspicious login detected. Verify identity now: {link} or account will be locked.",
    "You are pre-approved for a loan of ${amount} with 0% interest! Apply instantly at {link}",
    "CONGRATULATIONS! Your number was randomly selected to win a {item}. Reply YES to claim your reward now.",
    "Your subscription payment of ${amount} failed. Update your card details urgently at {link}",
    "Act now! Only 2 hours left to claim your free {item}. Click here: {link}",
    "IRS Notice: You have a pending refund of ${amount}. Submit your bank info at {link} to receive it.",
    "Hey it's {name}, I'm stuck abroad and need ${amount} urgently. Please send via {link}, will pay back!",
    "Your {service} account has been compromised. Reset your password immediately: {link}",
    "We tried to deliver your parcel but failed. Reschedule and pay ${amount} redelivery fee: {link}",
    "You've won a lottery of ${amount}! Send your bank details to claim your winnings today.",
    "SECURITY ALERT: Unusual activity detected on your account. Verify now at {link} to avoid suspension.",
    "Limited slots! Get {amount}% cashback instantly when you register your card at {link}",
    "This is your last chance to claim your ${amount} government stimulus. Apply here: {link}",
    "Your {service} password expires today. Click {link} immediately to avoid being locked out permanently.",
    "Job offer: Earn ${amount}/day working from home, no experience needed. Sign up now: {link}",
    "Your device has a virus! Download {item} now at {link} to protect your data before it's too late.",
]

# ---------------------------------------------------------------------------
# SUSPICIOUS messages (not outright fraud, but pushy/unverified/borderline)
# ---------------------------------------------------------------------------
suspicious_templates = [
    "Hi, I saw your profile and think you'd be a great fit for a part-time role. Interested? Reply for details.",
    "Reminder: your {service} trial ends soon. Upgrade now to keep your benefits, offer valid this week only.",
    "Hey, long time no talk! I switched numbers, can you send me your OTP so I can verify it's really you?",
    "We noticed you haven't logged in recently. Click here to reactivate your rewards points: {link}",
    "Special discount just for you: {amount}% off your next order, use code SAVE{amount} at checkout.",
    "Is this {name}? I have a business proposal that could earn you extra income, let's talk when free.",
    "Your feedback is needed! Complete this short survey and get a chance to win a gift card: {link}",
    "Hello, this is {bank} customer care. Can you confirm your last transaction amount for verification?",
    "New message request from an unknown number regarding a delivery you may have missed.",
    "Don't miss out! Flash sale ends tonight, grab your {item} before stock runs out: {link}",
    "Hi, I'm from HR, we'd like to schedule an interview. Please share your ID number for verification.",
    "Your friend invited you to try {service}. Sign up now and both of you get a bonus.",
    "We're updating our records, please confirm your date of birth and address at your earliest convenience.",
    "Hey, it's been a while. Can you help me out with something small? Call me when you get a chance.",
    "Your {service} membership is expiring. Renew today to avoid interruption of service.",
]

# ---------------------------------------------------------------------------
# SAFE messages (normal everyday conversation, legitimate notifications)
# ---------------------------------------------------------------------------
safe_templates = [
    "Hey, are we still meeting for lunch at {time} tomorrow?",
    "Your order #{digits} has been shipped and will arrive by {time}.",
    "Don't forget the team meeting today at {time} in the main conference room.",
    "Happy birthday! Hope you have a wonderful day 🎉",
    "Can you pick up milk on your way home?",
    "Reminder: your dentist appointment is scheduled for {time} on Friday.",
    "Thanks for coming to the party last night, it was great seeing you!",
    "The homework for tomorrow's class covers chapters 4 and 5.",
    "Your OTP for logging into your account is {digits}. Do not share this with anyone.",
    "Hi mom, landed safely, will call you once I get to the hotel.",
    "Just checking in, how did the exam go today?",
    "Your electricity bill of ${amount} is due on the 15th. Pay via the official app or website.",
    "Let's catch up this weekend, are you free on Saturday?",
    "The package you ordered has been delivered to your doorstep.",
    "Meeting rescheduled to {time}, please update your calendar.",
    "Congrats on the new job! We should celebrate soon.",
    "Can you send me the notes from today's lecture when you get a chance?",
    "Your flight {name} is confirmed for departure at {time}.",
    "Hey, dinner's ready whenever you get home.",
    "Your monthly statement is now available in your online banking portal.",
    "Good morning! Don't forget to bring your laptop charger to class.",
    "The weather looks great this weekend, want to go hiking?",
    "Your prescription is ready for pickup at the pharmacy.",
    "Thanks for the help moving apartments, couldn't have done it without you.",
]

links = ["bit.ly/claim-now", "secure-verify-account.com", "tinyurl.com/prize-win",
         "account-update-portal.net", "www.reward-center.co", "verify-now-safe.info"]
items = ["iPhone 16", "gift card", "laptop", "PlayStation 5", "smartwatch", "voucher"]
banks = ["Chase Bank", "HSBC", "Bank of America", "your bank", "Wells Fargo", "the bank"]
services = ["Netflix", "Amazon Prime", "email", "PayPal", "Instagram", "bank"]
names = ["John", "Sarah", "Alex", "Maria", "David", "Priya", "Ahmed", "Emma"]

def fill(t):
    return t.format(
        amount=random.choice([50, 100, 150, 200, 250, 300, 500, 750, 1000, 1500,
                               2000, 2500, 3000, 5000, 7500, 10000, 10, 15, 20, 25, 30, 40]),
        link=random.choice(links),
        item=random.choice(items),
        digits=str(random.randint(1000, 999999)),
        bank=random.choice(banks),
        name=random.choice(names),
        service=random.choice(services),
        time=random.choice(["8 AM", "9 AM", "10 AM", "11 AM", "noon", "1 PM", "2 PM",
                             "3 PM", "4 PM", "5 PM", "6 PM", "6:30 PM", "7 PM", "8 PM", "9 PM"]),
    )

rows = []

def generate(templates, label, multiplier):
    for _ in range(multiplier):
        for t in templates:
            rows.append((fill(t), label))

generate(scam_templates, "SCAM", 11)
generate(suspicious_templates, "SUSPICIOUS", 26)
generate(safe_templates, "SAFE", 22)

random.shuffle(rows)

# De-duplicate while keeping order (some fills may coincide)
seen = set()
unique_rows = []
for text, label in rows:
    key = text.strip().lower()
    if key not in seen:
        seen.add(key)
        unique_rows.append((text, label))

with open("/home/claude/scam_detector/data/scam_dataset.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["message", "label"])
    writer.writerows(unique_rows)

print(f"Generated {len(unique_rows)} unique labeled messages.")
from collections import Counter
print(Counter(label for _, label in unique_rows))
