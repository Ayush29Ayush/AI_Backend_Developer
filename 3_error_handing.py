from openai import OpenAI, APIError, RateLimitError, APITimeoutError
import time

client = OpenAI()

def call_llm_with_retry(messages, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                timeout=30.0  # 30 second timeout
            )
            return response.choices[0].message.content

        except RateLimitError:
            # Wait and retry
            wait_time = 2 ** attempt  # Exponential backoff
            print(f"Rate limited. Waiting {wait_time}s...")
            time.sleep(wait_time)

        except APITimeoutError:
            print("Request timed out. Retrying...")
            continue

        except APIError as e:
            print(f"API error: {e}")
            raise

    raise Exception("Max retries exceeded")