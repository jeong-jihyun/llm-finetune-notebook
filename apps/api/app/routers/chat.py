from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.auth import require_password
from app.services import rag

router = APIRouter(prefix="/chat", tags=["chat"], dependencies=[Depends(require_password)])


class ChatIn(BaseModel):
    question: str


@router.post("")
def chat(body: ChatIn):
    q = body.question.strip()
    if not q:
        raise HTTPException(400, "질문을 입력해 주세요")
    return rag.answer(q)
