from fastapi import FastAPI
from pydantic import BaseModel
from pinecone import Pinecone, ServerlessSpec
import openai
import os

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

openai.api_key = OPENAI_API_KEY
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("aether-core")

app = FastAPI()

class Item(BaseModel):
    id: str
    text: str

class UpsertRequest(BaseModel):
    namespace: str
    items: list[Item]

@app.post("/upsert")
async def upsert_vectors(data: UpsertRequest):
    texts = [item.text for item in data.items]
    embeddings = openai.embeddings.create(input=texts, model="text-embedding-3-small")
    vectors = [
        {
            "id": item.id,
            "values": embeddings.data[i].embedding,
            "metadata": {"text": item.text}
        }
        for i, item in enumerate(data.items)
    ]
    index.upsert(vectors=vectors, namespace=data.namespace)
    return {"status": "success", "count": len(vectors)}
