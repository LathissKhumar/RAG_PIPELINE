from fastapi import APIRouter
from app.utils.rag_pipeline import rag_query

router = APIRouter()

@router.get("/ask")
def ask_q(question: str):
    answer = rag_query(question)
    return {"question": question, "answer": answer}
