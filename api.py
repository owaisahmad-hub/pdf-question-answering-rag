from fastapi import FastAPI
from pydantic import BaseModel
from ask import ask

app = FastAPI(title="PDF Q&A with citations")

class Question(BaseModel):
    question: str

@app.post("/ask")
def ask_endpoint(q: Question):
    answer, sources = ask(q.question)
    return {"answer": answer, "sources": sources}