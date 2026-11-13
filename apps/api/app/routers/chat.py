from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services import rag

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatIn(BaseModel):
    question: str
    model: str | None = None  # 2단계: 학습된 모델 ID


@router.post("")
def chat(body: ChatIn):
    q = body.question.strip()
    if not q:
        raise HTTPException(400, "질문을 입력해 주세요")
    return rag.answer(q, body.model)
