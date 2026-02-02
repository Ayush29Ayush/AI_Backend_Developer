from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

#! Bad System Prompt
# response = client.chat.completions.create(
#     model="gpt-4o-mini",
#     messages = [
#     {"role": "system", "content": "You are helpful."},
#     {"role": "user", "content": "Summarize this article..."}
# ]
# )

#! Good System Prompt

SYSTEM_PROMPT = """You are a content summarizer for a tech news app.

TASK: Summarize articles for busy developers.

RULES:
- Maximum 3 bullet points
- Each bullet under 20 words
- Focus on practical implications, not hype
- If the article is not tech-related, respond with "NOT_RELEVANT"

OUTPUT FORMAT:
- Title: <Concise title>
- Bullet 1
- Bullet 2
- Bullet 3
"""

ARTICLE = """Indian tech startup XYZ has launched a new AI-powered coding assistant aimed at helping developers write code faster and with fewer errors. The tool integrates with popular IDEs and supports multiple programming languages. Early users report significant improvements in productivity and code quality."""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": f"Summarize this article: {ARTICLE}"}
]
)

print(response.choices[0].message.content)