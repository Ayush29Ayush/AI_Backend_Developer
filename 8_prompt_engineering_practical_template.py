"""
Structured Action Item Extractor
---------------------------------
Creates a structured system prompt and sends it to OpenAI
to extract action items from meeting notes as valid JSON.
"""

import os
import json
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI


# -----------------------------
# Configuration
# -----------------------------

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise EnvironmentError("OPENAI_API_KEY not found in environment variables.")

client = OpenAI(api_key=API_KEY)


# -----------------------------
# Prompt Builder
# -----------------------------

def create_system_prompt(task: str, output_schema: Dict[str, Any], rules: List[str]) -> str:
    """
    Creates a structured system prompt for controlled LLM output.

    :param task: Description of the task
    :param output_schema: Expected JSON schema
    :param rules: List of behavioral rules
    :return: Formatted system prompt string
    """

    rules_text = "\n".join(f"- {rule}" for rule in rules)

    return f"""You are a backend service component.

TASK:
{task}

OUTPUT:
Return valid JSON matching this schema:
{json.dumps(output_schema, indent=2)}

RULES:
{rules_text}

If you cannot complete the task, return:
{{"error": "reason"}}
"""


# -----------------------------
# LLM Request Function
# -----------------------------

def extract_action_items(meeting_notes: str) -> Dict[str, Any]:
    """
    Sends meeting notes to OpenAI and extracts structured action items.

    :param meeting_notes: Raw meeting transcript
    :return: Parsed JSON response
    """

    system_prompt = create_system_prompt(
        task="Extract action items from meeting notes",
        output_schema={
            "action_items": [
                {"task": "string", "assignee": "string", "due": "string"}
            ]
        },
        rules=[
            "Only include explicit action items, not general discussion",
            "If no assignee mentioned, use 'unassigned'",
            "If no due date mentioned, use 'no date'",
            "Return ONLY valid JSON. No explanation text."
        ],
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": meeting_notes},
        ],
        response_format={"type": "json_object"},  # Enforces JSON output
    )

    return json.loads(response.choices[0].message.content)


# -----------------------------
# Example Usage
# -----------------------------

#! Ensures this block runs only when the file is executed directly, not when imported as a module.
if __name__ == "__main__":

    SAMPLE_MEETING_NOTES = """
    John will prepare the quarterly budget report by Friday.
    Sarah needs to update the landing page design.
    We discussed improving customer onboarding.
    Mike will schedule the client demo next Tuesday.
    """

    try:
        result = extract_action_items(SAMPLE_MEETING_NOTES)
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(json.dumps({"error": str(e)}, indent=2))