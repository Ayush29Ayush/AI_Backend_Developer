from openai import OpenAI
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)

# Connect to your Postgres database
conn = psycopg2.connect("postgresql://postgres:postgres@localhost:5433/ai_backend_database")

#! Frequently Asked Questions (FAQ) Search Example
class FAQSearch:
    def __init__(self):
        self._ensure_table()

    def _ensure_table(self):
        """Create the faqs table if it doesn't exist."""
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS faqs (
                        id SERIAL PRIMARY KEY,
                        question TEXT UNIQUE,
                        answer TEXT,
                        embedding vector(3072)
                    )
                """)
            conn.commit()
        except Exception as e:
            print("Error creating table:", e)

    def add_faq(self, question: str, answer: str):
        """Add a FAQ with embedding, using UPSERT to avoid duplicates."""
        try:
            embedding = self._get_embedding(question)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO faqs (question, answer, embedding)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (question)
                    DO UPDATE SET embedding = EXCLUDED.embedding, answer = EXCLUDED.answer
                    """,
                    (question, answer, embedding)
                )
            conn.commit()
            print(f"Stored FAQ: {question[:50]}...")
        except Exception as e:
            print("Error storing FAQ:", e)

    def search(self, query: str, threshold: float = 0.2):
        """Search for the most similar FAQ above a similarity threshold."""
        try:
            query_embedding = self._get_embedding(query)
        except Exception as e:
            print("Error generating query embedding:", e)
            return None

        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT question, answer, 1 - (embedding <=> %s::vector) AS similarity
                    FROM faqs
                    WHERE 1 - (embedding <=> %s::vector) > %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT 1
                """, (query_embedding, query_embedding, threshold, query_embedding))
                row = cur.fetchone()
                if row:
                    return {"question": row[0], "answer": row[1], "similarity": round(row[2], 3)}
        except Exception as e:
            print("Error querying FAQs:", e)
        return None

    def _get_embedding(self, text: str):
        """Generate embedding using text-embedding-3-large."""
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=text
        )
        return response.data[0].embedding


# -----------------------------
# Example Usage
# -----------------------------
faq = FAQSearch()

# Add some FAQs
faq.add_faq("How do I cancel my subscription?", "Go to Settings > Billing > Cancel")
faq.add_faq("What payment methods do you accept?", "We accept Visa, Mastercard, PayPal")

# Search
query = "I want to stop paying"
result = faq.search(query)

if result:
    print(f"Q: {result['question']}")
    print(f"A: {result['answer']}")
    print(f"Confidence: {result['similarity']:.0%}")
else:
    print("No FAQ matched your query.")