from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

#! Zero-Shot Prompting
# Just ask the model to do something without providing examples.

messages = [
    {
        "role": "system",
        "content": "Convert natural language to SQL."
    },
    {
        "role": "user",
        "content": "Show me all users who signed up last week"
    }
]
# Output might be inconsistent

def zero_shot_openai(user_query: str) -> dict:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Convert natural language to SQL for a users table.\n"
                    "Table schema: users(id, email, name, created_at, plan)\n\n"
                    "Rules:\n"
                    "- Return ONLY valid SQL\n"
                    "- No explanations, no formatting"
                ),
            },
            {
                "role": "user",
                "content": user_query,
            },
        ],
    )
    return response

result = zero_shot_openai("Show me all premium users")
print(result.choices[0].message.content)
# SELECT * FROM users WHERE plan = 'premium';

#! Few-Shot Prompting
# Show the model examples of what you want. This is the most effective technique for consistent outputs.

def few_shot_openai(user_query: str) -> dict:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Convert natural language to SQL for a users table.\n"
                    "Table schema: users(id, email, name, created_at, plan)\n\n"
                    "Rules:\n"
                    "- Return ONLY valid SQL\n"
                    "- No explanations, no formatting"
                ),
            },
            # Example 1
            {
                "role": "user",
                "content": "Get all premium users",
            },
            {
                "role": "assistant",
                "content": "SELECT * FROM users WHERE plan = 'premium';",
            },
            # Example 2
            {
                "role": "user",
                "content": "Count users by plan",
            },
            {
                "role": "assistant",
                "content": "SELECT plan, COUNT(*) FROM users GROUP BY plan;",
            },
            # Actual query
            {
                "role": "user",
                "content": user_query,
            },
        ],
    )
    return response

result = few_shot_openai("Show me all users who signed up last week")
print(result.choices[0].message.content)
# SELECT * FROM users WHERE created_at >= NOW() - INTERVAL '7 days';

# --------------------------------------------------------------------

#! Summary
# Zero-shot prompting => No examples. Works for simple tasks.
# Few-shot prompting => 2-5 examples. Best for structured outputs.
# Many-shot prompting => 10+ examples. When precision is critical.
