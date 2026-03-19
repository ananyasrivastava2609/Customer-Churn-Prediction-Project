"""
Generate reproducible demo CSVs: sample_churn.csv and sample_tickets.csv.
No PII; placeholder IDs only. Run from project root: python data/generate_demo_data.py
"""
import random
from pathlib import Path
import pandas as pd

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

N_CHURN = 800
N_TICKETS = 500


def generate_churn():
    ages = list(range(22, 70))
    genders = ["M", "F", "Other"]
    plans = ["basic", "premium", "enterprise"]
    rows = []
    for i in range(N_CHURN):
        churn = 1 if random.random() < 0.25 else 0
        tenure = random.randint(1, 60)
        usage = random.uniform(5, 120) if not churn else random.uniform(1, 40)
        support = random.randint(0, 15) if churn else random.randint(0, 5)
        delay = random.randint(0, 30) if churn else random.randint(0, 7)
        rows.append({
            "customer_id": f"cust_{i+1000}",
            "age": random.choice(ages),
            "gender": random.choice(genders),
            "tenure_months": tenure,
            "monthly_usage_hours": round(usage, 2),
            "support_calls": support,
            "payment_delay_days": delay,
            "subscription_type": random.choice(plans),
            "contract_length_months": random.choice([12, 24, 36]),
            "total_spend": round(random.uniform(100, 5000), 2),
            "last_interaction_date": "2024-01-15",
            "churn": "Yes" if churn else "No",
        })
    df = pd.DataFrame(rows)
    path = Path(__file__).parent / "sample_churn.csv"
    df.to_csv(path, index=False)
    print(f"Wrote {path} ({len(df)} rows).")
    return path


def generate_tickets():
    subjects = [
        "Billing question", "Login failed", "Slow performance", "Refund request",
        "How to export data", "Urgent: service down", "Complaint about support",
        "Feature request", "Password reset", "API error 500",
    ]
    bodies = [
        "I have a question about my invoice.",
        "I cannot log in. Getting error message.",
        "The app is very slow lately.",
        "I would like to request a full refund. Please cancel my subscription.",
        "How do I export my data to CSV?",
        "Our whole team cannot access the service. This is critical.",
        "Very disappointed with the support response. Need escalation.",
        "It would be great to have dark mode.",
        "I forgot my password.",
        "We are getting 500 errors when calling the API.",
    ]
    priorities = ["low", "medium", "high", "critical"]
    rows = []
    for i in range(N_TICKETS):
        j = random.randint(0, len(subjects) - 1)
        subj = subjects[j]
        body = bodies[j] + f" Ticket ref #{i+1}."
        # Derive priority from content for realism
        if "urgent" in subj.lower() or "down" in body.lower() or "refund" in body.lower():
            pri = "critical"
        elif "failed" in subj.lower() or "error" in body.lower() or "complaint" in body.lower():
            pri = "high"
        elif "question" in subj.lower() or "how" in body.lower():
            pri = "medium"
        else:
            pri = "low"
        rows.append({
            "ticket_id": f"T{i+100}",
            "customer_id": f"cust_{random.randint(1000, 1000 + N_CHURN)}",
            "ticket_subject": subj,
            "ticket_description": body,
            "ticket_priority": pri,
        })
    df = pd.DataFrame(rows)
    path = Path(__file__).parent / "sample_tickets.csv"
    df.to_csv(path, index=False)
    print(f"Wrote {path} ({len(df)} rows).")
    return path


if __name__ == "__main__":
    Path(__file__).parent.mkdir(parents=True, exist_ok=True)
    generate_churn()
    generate_tickets()
