from openai import OpenAI
from dotenv import load_dotenv
import json

load_dotenv()
client = OpenAI()

#! When using JSON mode, you MUST mention "JSON" in your system prompt. Otherwise OpenAI may return an error.
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "system",
            "content": """Extract product info from descriptions.
            Return JSON with: name, price, category, in_stock (boolean)"""
        },
        {
            "role": "user",
            "content": "The Nike Air Max 90 is available now for $150"
        }
    ],
    response_format={"type": "json_object"}  # Forces valid JSON
)

data = json.loads(response.choices[0].message.content)
print(data)
# {"name": "Nike Air Max 90", "price": 150, "category": "shoes", "in_stock": true}