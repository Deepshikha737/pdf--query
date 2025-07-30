# main.py

from fastapi import FastAPI, UploadFile, Form
from app.parser import extract_text_from_pdf
from app.chunker import chunk_text
from app.retriever import store_chunks_in_pinecone, query_chunks_from_pinecone
from app.groq_llm import query_groq_llm

  # updated import
import uuid
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

@app.post("/run")
async def run_query(file: UploadFile, question: str = Form(...)):
    file_bytes = await file.read()
    raw_text = extract_text_from_pdf(file_bytes)

    chunks = chunk_text(raw_text)
    file_id = str(uuid.uuid4())

    store_chunks_in_pinecone(chunks, file_id)

    top_chunks = query_chunks_from_pinecone(question)
    answer = query_groq_llm(" ".join(top_chunks), question)  # updated call

    return {
        "question": question,
        "context_used": top_chunks,
        "answer": answer
    }

@app.get("/")
def read_root():
    return {"message": "LLM PDF QA API is running. Go to /docs for Swagger UI."}
