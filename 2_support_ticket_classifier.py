from dotenv import load_dotenv
from openai import OpenAI
import json

load_dotenv()
client = OpenAI()

def classify_ticket(ticket_text: str) -> dict:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,  # Deterministic for classification
        messages=[
            {
                "role": "system",
                "content": """Classify support tickets into categories.
                Return JSON with: category, priority, summary.
                Categories: billing, technical, account, general.
                Priorities: low, medium, high, urgent"""
            },
            {
                "role": "user",
                "content": ticket_text
            }
        ],
        response_format={"type": "json_object"}  # Ensures valid JSON
    )

    return json.loads(response.choices[0].message.content)

# Test it
ticket = "I've been charged twice for my subscription this month!"
result = classify_ticket(ticket)
print(result)
# {'category': 'billing', 'priority': 'high', 'summary': 'Customer has been charged twice for their subscription.'}