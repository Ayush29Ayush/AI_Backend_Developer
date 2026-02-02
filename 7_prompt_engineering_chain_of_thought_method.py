from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "system",
            "content": (
                "You are a code reviewer. When reviewing code:\n"
                "1. First, identify what the code is trying to do\n"
                "2. Then, list any bugs or issues\n"
                "3. Then, suggest improvements\n"
                "4. Finally, give an overall rating (1-5)\n\n"
                "Think through each step before giving your final answer."
            ),
        },
        {
            "role": "user",
            "content": (
                "Review this code:\n"
                "def get_user(id):\n"
                "    user = db.query(f\"SELECT * FROM users WHERE id = {id}\")\n"
                "    return user"
            ),
        },
    ],
)

review = response.choices[0].message.content
print(review)

# Expected output:
"""
1. Identifying what the code is trying to do:
   The function `get_user(id)` is intended to retrieve a user record from a
   database based on the provided `id`. It executes a query to select all
   columns from the `users` table where the id matches the input parameter
   and returns the result.

2. Bugs or issues:
   - SQL Injection Risk: The `id` is directly interpolated into the SQL
     query string, making the function vulnerable to SQL injection.
   - Missing Error Handling: Database failures are not handled gracefully.
   - Return Type Ambiguity: It is unclear whether the query returns a single
     user, multiple users, or none.

3. Suggested improvements:
   - Use parameterized queries to prevent SQL injection.
   - Add proper error handling around the database call.
   - Clarify the return value (e.g., return a single user or None).

   Example improvement:

       def get_user(id):
           try:
               query = "SELECT * FROM users WHERE id = ?"
               user = db.query(query, (id,))
               if user:
                   return user[0]
               return None
           except Exception as e:
               print(f"An error occurred: {e}")
               return None

4. Overall rating (1-5):
   Rating: 2
"""

