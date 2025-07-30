import os
from pinecone import Pinecone
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# Get Pinecone config from environment
api_key = os.getenv("PINECONE_API_KEY")
index_name = os.getenv("PINECONE_INDEX_NAME")
region = os.getenv("PINECONE_ENVIRONMENT")

if not all([api_key, index_name, region]):
    raise EnvironmentError("Missing one or more Pinecone environment variables.")

# Initialize Pinecone client
pc = Pinecone(api_key=api_key)
index = pc.Index(index_name=index_name, region=region)

# Lazy-load the SentenceTransformer to reduce memory use on startup
@lru_cache()
def get_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

def store_chunks_in_pinecone(chunks, file_id):
    embedder = get_embedder()
    vectors = [
        {
            "id": f"{file_id}-{i}",
            "values": embedder.encode(chunk).tolist(),
            "metadata": {"text": chunk}
        }
        for i, chunk in enumerate(chunks)
    ]
    index.upsert(vectors=vectors)

def query_chunks_from_pinecone(query, top_k=5):
    embedder = get_embedder()
    query_vec = embedder.encode(query).tolist()
    results = index.query(vector=query_vec, top_k=top_k, include_metadata=True)
    return [match["metadata"]["text"] for match in results["matches"]]
