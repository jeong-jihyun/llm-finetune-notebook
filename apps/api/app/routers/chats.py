from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.auth import require_password
from app.services import chats, rag, search as search_svc

router = APIRouter(prefix="/chats", tags=["chats"], dependencies=[Depends(require_password)])
search_router = APIRouter(prefix="/search", tags=["search"], dependencies=[Depends(require_password)])


class TitleIn(BaseModel):
    title: str = ""


class AskIn(BaseModel):
    question: str


@router.get("")
def list_groups():
    return chats.list_groups()


@router.post("")
def create(body: TitleIn):
    return chats.create_group(body.title)


@router.patch("/{gid}")
def rename(gid: str, body: TitleIn):
    return chats.rename_group(gid, body.title)


@router.delete("/{gid}")
def delete(gid: str):
    chats.delete_group(gid)
    return {"ok": True}


@router.get("/{gid}")
def messages(gid: str):
    return {"id": gid, "messages": chats.get_messages(gid)}


@router.post("/{gid}/ask")
def ask(gid: str, body: AskIn):
    q = body.question.strip()
    if not q:
        raise HTTPException(400, "질문을 입력해 주세요")
    chats.get_messages(gid)  # 그룹이 있는지 먼저 확인
    result = rag.answer(q)
    chats.add_exchange(gid, q, result)  # 답변이 성공했을 때만 기록
    return result


@search_router.get("")
def search(q: str = ""):
    return search_svc.search(q.strip())
