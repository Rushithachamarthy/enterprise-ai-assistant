# vector_store.py
# Handles text splitting and vector search

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load embedding model once
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def chunk_text(text, chunk_size=500, overlap=100):
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks if chunks else [""]

def create_vector_store(text):
    chunks = chunk_text(text)
    if not chunks or not chunks[0]:
        return None, []

    embeddings = embedding_model.encode(chunks, normalize_embeddings=True)
    embeddings = embeddings.astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index, chunks

def retrieve_chunks(index, chunks, query, top_k=15):
    if index is None or not chunks:
        return []

    query_emb = embedding_model.encode([query], normalize_embeddings=True).astype("float32")
    D, I = index.search(query_emb, top_k)
    
    retrieved = []
    for i in I[0]:
        if i != -1 and i < len(chunks):
            retrieved.append(chunks[i])
    return retrieved