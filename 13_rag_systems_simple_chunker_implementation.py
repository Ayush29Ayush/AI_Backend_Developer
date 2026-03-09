# ------------------------------------------------------------
# TEXT CHUNKING FUNCTION NOTES
# ------------------------------------------------------------

# Purpose:
# This function splits a long text into smaller overlapping chunks.
# It is useful when working with large documents that need to be
# processed in parts (e.g., NLP, embeddings, LLMs, search systems).

# Function Parameters:
# text (str)        -> The input text that needs to be split.
# chunk_size (int)  -> Number of words in each chunk (default = 500).
# overlap (int)     -> Number of words shared between consecutive chunks to maintain context (default = 50).

# Return:
# Returns a list of strings where each string is a chunk of the original text.

# ------------------------------------------------------------
# HOW THE FUNCTION WORKS
# ------------------------------------------------------------

# 1. The text is first split into a list of words using text.split().
#    Example:
#    "I love machine learning"
#    -> ["I", "love", "machine", "learning"]

# 2. A loop starts from the first word and keeps creating chunks
#    until all words are processed.

# 3. For each iteration:
#    start -> starting index of the chunk
#    end   -> start + chunk_size

# 4. Words from start to end are joined back into a string
#    using ' '.join() to form a chunk.

# 5. The chunk is added to the chunks list.

# 6. Instead of moving directly to the next chunk, we move back by 'overlap' words:
#
#       start = end - overlap
#
#    This creates overlapping chunks so context is preserved.

# ------------------------------------------------------------
# EXAMPLE
# ------------------------------------------------------------

# Text:
# "The quick brown fox jumps over the lazy dog"

# chunk_size = 4
# overlap = 1

# Chunk 1:
# "The quick brown fox"

# Chunk 2:
# "fox jumps over the"

# Chunk 3:
# "the lazy dog"

# Notice that some words repeat between chunks because of overlap.

# ------------------------------------------------------------
# WHY OVERLAP IS IMPORTANT
# ------------------------------------------------------------

# Overlap helps preserve context between chunks.
# Without overlap, important information might be lost
# between two chunks.

# Example without overlap:
# Chunk1: "machine learning is"
# Chunk2: "very powerful"

# Example with overlap:
# Chunk1: "machine learning is"
# Chunk2: "learning is very"
# Chunk3: "is very powerful"

# This helps models understand continuity.

# ------------------------------------------------------------
# COMMON USE CASES
# ------------------------------------------------------------

# 1. Retrieval Augmented Generation (RAG)
# 2. Text embeddings
# 3. Document search systems
# 4. Summarizing long documents
# 5. Processing large PDFs or books
# ------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        start = end - overlap  # Overlap for context continuity

    return chunks

# Example
document = "Your very long document text here..."
chunks = chunk_text(document, chunk_size=300, overlap=30)
print(f"Created {len(chunks)} chunks")