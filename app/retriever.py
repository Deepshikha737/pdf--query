import os
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

def get_index():
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    return pc.Index(os.getenv("PINECONE_INDEX_NAME"))

def get_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

def store_chunks_in_pinecone(chunks, file_id):
    embedder = get_embedder()
    index = get_index()
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
    index = get_index()
    query_vec = embedder.encode(query).tolist()
    results = index.query(vector=query_vec, top_k=top_k, include_metadata=True)
    return [match["metadata"]["text"] for match in results["matches"]]
