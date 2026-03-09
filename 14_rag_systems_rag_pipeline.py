
# Document
#    ↓
# Chunking
#    ↓
# Embedding
#    ↓
# UPSERT (dedupe)
#    ↓
# pgvector storage
#    ↓
# Vector index search
#    ↓
# Top chunks
#    ↓
# GPT answer

from openai import OpenAI
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)

# Database connection (same style as your previous code)
conn = psycopg2.connect("postgresql://postgres:postgres@localhost:5433/ai_backend_database")


class RAGPipeline:

    def __init__(self):
        self._ensure_table()

    def _ensure_table(self):
        """Create chunks table and index if they don't exist."""
        try:
            with conn.cursor() as cur:

                # Create table with UNIQUE constraint
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS chunks (
                        id SERIAL PRIMARY KEY,
                        content TEXT,
                        source TEXT,
                        embedding vector(3072),
                        UNIQUE(content, source)
                    )
                """)

                # Create pgvector similarity index
                # Index commented out because IVFFLAT only supports vectors ≤ 2000 dimensions, and 3072-dim embeddings would fail
                # cur.execute("""
                #     CREATE INDEX IF NOT EXISTS chunks_embedding_idx
                #     ON chunks
                #     USING ivfflat (embedding vector_cosine_ops)
                #     WITH (lists = 100)
                # """)

            conn.commit()

        except Exception as e:
            print("Error creating table or index:", e)

    # -----------------------------
    # Document ingestion
    # -----------------------------
    def ingest_document(self, content: str, source: str):
        """Chunk and store document in DB."""
        try:
            chunks = self._chunk_text(content)

            with conn.cursor() as cur:
                for chunk in chunks:
                    embedding = self._get_embedding(chunk)

                    cur.execute(
                        """
                        INSERT INTO chunks (content, source, embedding)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (content, source)
                        DO UPDATE SET embedding = EXCLUDED.embedding
                        """,
                        (chunk, source, embedding)
                    )

            conn.commit()
            print(f"Processed {len(chunks)} chunks from {source}")

        except Exception as e:
            print("Error ingesting document:", e)

    # -----------------------------
    # Query pipeline
    # -----------------------------
    def query(self, question: str, top_k: int = 3):

        context_chunks = self._retrieve(question, top_k)

        if not context_chunks:
            return "I don't have information about that."

        context = "\n\n".join([c["content"] for c in context_chunks])
        sources = list(set([c["source"] for c in context_chunks]))

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": """
                        Answer questions using ONLY the provided context.
                        If the answer is not in the context say:
                        'I don't have information about that.'
                        """
                    },
                    {
                        "role": "user",
                        "content": f"""
Context:
{context}

Question: {question}
"""
                    }
                ]
            )

            answer = response.choices[0].message.content

            return f"{answer}\n\nSources: {', '.join(sources)}"

        except Exception as e:
            print("Error generating answer:", e)
            return None

    # -----------------------------
    # Vector search
    # -----------------------------
    def _retrieve(self, query: str, top_k: int):

        try:
            query_embedding = self._get_embedding(query)

            with conn.cursor() as cur:
                cur.execute("""
                    SELECT content, source,
                    1 - (embedding <=> %s::vector) AS similarity
                    FROM chunks
                    WHERE 1 - (embedding <=> %s::vector) > 0.4
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """, (query_embedding, query_embedding, query_embedding, top_k))

                rows = cur.fetchall()

            return [
                {
                    "content": r[0],
                    "source": r[1],
                    "similarity": round(r[2], 3)
                }
                for r in rows
            ]

        except Exception as e:
            print("Error retrieving chunks:", e)
            return []

    # -----------------------------
    # Chunking
    # -----------------------------
    def _chunk_text(self, text: str, size: int = 300, overlap: int = 30):

        words = text.split()
        chunks = []
        start = 0

        while start < len(words):
            chunk = " ".join(words[start:start + size])
            chunks.append(chunk)
            start += size - overlap

        return chunks

    # -----------------------------
    # Embeddings
    # -----------------------------
    def _get_embedding(self, text: str):

        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=text
        )

        return response.data[0].embedding


# -----------------------------
# Example Usage
# -----------------------------

# Initialize RAG pipeline
rag = RAGPipeline()

document = """
Artificial Intelligence is a field of computer science that focuses on
creating machines capable of performing tasks that typically require
human intelligence such as learning, reasoning, and decision making.
"""

rag.ingest_document(document, "AI_Article")

# Ingest documents
rag.ingest_document("""
Our refund policy: We offer a 14-day money-back guarantee for all annual plans.
Monthly plans can be cancelled anytime but are not refundable.
To request a refund, email support@example.com with your order ID.
""", source="refund-policy.md")

rag.ingest_document("""
Pricing Plans:
- Starter: $9/month or $90/year (save $18)
- Pro: $29/month or $290/year (save $58)
- Enterprise: Custom pricing, contact sales
All plans include 14-day free trial.
""", source="pricing.md")

# answer = rag.query("What is artificial intelligence?")

# print("\nAnswer:")
# print(answer)

# Query
answer = rag.query("Can I get my money back if I don't like the product?")
print("\nAnswer:")
print(answer)

# Expected Output Example:
# Answer:
# It depends on the plan: if you are on an annual plan, you can get a refund within 14 days. However, if you are on a monthly plan, it is not refundable.

# Sources: refund-policy.md