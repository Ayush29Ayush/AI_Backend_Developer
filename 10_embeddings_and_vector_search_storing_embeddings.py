
#! Install the Extension
# -- In your Postgres database
# CREATE EXTENSION IF NOT EXISTS vector;

#! Create a Table with Vector Column
# CREATE TABLE documents (
#     id SERIAL PRIMARY KEY,
#     content TEXT NOT NULL,
#     embedding vector(1536),  -- 1536 dimensions for OpenAI
#     created_at TIMESTAMP DEFAULT NOW()
# );

# -- Create an index for fast similarity search
# CREATE INDEX ON documents USING ivfflat (embedding vector_cosine_ops)
# WITH (lists = 100);

#! Storing Embeddings
import psycopg2
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)

conn = psycopg2.connect("postgresql://postgres:postgres@localhost:5433/ai_backend_database")

def store_document(content: str):
    # Generate embedding
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=content
    )
    embedding = response.data[0].embedding

    # Store in Postgres
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO documents (content, embedding) VALUES (%s, %s)",
            (content, embedding)
        )
    conn.commit()

# Store some documents
store_document("How to reset your password: Go to Settings > Security > Reset Password")
store_document("Changing your email: Navigate to Profile > Edit > Email Address")
store_document("Billing FAQ: We accept Visa, Mastercard, and PayPal")