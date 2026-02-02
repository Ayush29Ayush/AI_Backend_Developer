from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()  # Automatically reads OPENAI_API_KEY

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is Python? Give a one line short answer."}]
)

print(response.choices[0].message.content)

print("----------------------------------------------------------------")

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hi!"}]
)

# The response object
print(response)                             # Full response object
print(response.choices[0].message.content)  # The actual text
print(response.choices[0].finish_reason)    # "stop" = completed normally
print(response.usage.prompt_tokens)         # Tokens in your request
print(response.usage.completion_tokens)     # Tokens in response
print(response.usage.total_tokens)          # Total (for billing)
