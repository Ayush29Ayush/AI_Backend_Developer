import psycopg2
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)

# Make sure your database name is correct
conn = psycopg2.connect("postgresql://postgres:postgres@localhost:5433/ai_backend_database")


def semantic_search(query: str, limit: int = 5) -> list[dict]:
    try:
        # Generate embedding for the query
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=query
        )
        query_embedding = response.data[0].embedding
    except Exception as e:
        print("Error generating embedding:", e)
        return []

    try:
        # Find similar documents
        with conn.cursor() as cur:
            cur.execute("""
                SELECT content, 1 - (embedding <=> %s::vector) AS similarity
                FROM documents
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (query_embedding, query_embedding, limit))

            rows = cur.fetchall()

        if not rows:
            print("No similar documents found.")
            return []

        # Build result list
        results = [
            {"content": row[0], "similarity": round(float(row[1]), 3)}
            for row in rows
        ]
        return results

    except Exception as e:
        print("Error querying database:", e)
        return []


# -----------------------
# Run Search
# -----------------------
query_text = "I forgot my login credentials, reset password"
results = semantic_search(query_text)
print(results)

if results:
    print(f"Top results for query: '{query_text}'\n")
    for r in results:
        print(f"{r['similarity']}: {r['content'][:60]}...")
else:
    print("No results found or an error occurred.")

#! Output example:
# Top results for query: 'I forgot my login credentials, reset password'

# 0.537: How to reset your password: Go to Settings > Security > Rese...
# 0.311: Changing your email: Navigate to Profile > Edit > Email Addr...
# 0.16: Billing FAQ: We accept Visa, Mastercard, and PayPal...

# ---------------------------------------------------------------------------------------

#! Distance Operator
# pgvector uses <=> for cosine distance:

# -- <=> returns cosine DISTANCE (lower = more similar)
# -- To get SIMILARITY (higher = more similar), use: 1 - distance

# SELECT content,
#        1 - (embedding <=> query_embedding) AS similarity
# FROM documents
# ORDER BY embedding <=> query_embedding  -- Closest first
# LIMIT 5;