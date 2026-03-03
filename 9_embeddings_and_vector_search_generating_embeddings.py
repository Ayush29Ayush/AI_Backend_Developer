from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

def get_embedding(text: str) -> list[float]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

# Generate an embedding
embedding = get_embedding("How do I reset my password?")
print(f"Dimensions: {len(embedding)}")  # 1536
print(f"First 5 values: {embedding[:5]}")  # [0.023, -0.045, ...]

# Model	                    Dimensions	             Best for
# text-embedding-3-small	1536	                 Most use cases, good balance
# text-embedding-3-large	3072	                 Higher accuracy, more storage