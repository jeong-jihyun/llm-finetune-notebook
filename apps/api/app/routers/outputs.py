from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.auth import require_password
from app.services import chats, generate, ingest, outputs

router = APIRouter(prefix="/outputs", tags=["outputs"], dependencies=[Depends(require_password)])


class PatchIn(BaseModel):
    title: str


class ContinueIn(BaseModel):
    instruction: str = ""


def _with_group(o: dict) -> dict:
    gid = o.get("group_id") or ""
    g = chats.get_group(gid) if gid and chats.exists(gid) else None
    return {**o, "group_title": g["title"] if g else ("(삭제된 대화)" if gid else ""), "group_alive": bool(g)}


@router.get("")
def list_all():
    return [_with_group(o) for o in outputs.list_outputs()]


@router.get("/{oid}")
def get_one(oid: str):
    return _with_group(outputs.get(oid))


@router.patch("/{oid}")
def rename(oid: str, body: PatchIn):
    return _with_group(outputs.update(oid, title=body.title))


@router.delete("/{oid}")
def delete(oid: str):
    outputs.delete(oid)
    return {"ok": True}


@router.post("/{oid}/continue")
def continue_writing(oid: str, body: ContinueIn):
    """✍ 이어서 쓰기: Gemini가 기존 글 뒤에 새 내용을 붙여 준다."""
    return _with_group(generate.continue_output(oid, body.instruction))


@router.post("/{oid}/open-chat")
def open_chat(oid: str):
    """💬 대화로 이어가기: 이 글을 자료로 넣은 대화방을 열어 준다 (이미 만들었으면 그 방을 다시 연다)."""
    o = outputs.get(oid)
    if o.get("chat_id") and chats.exists(o["chat_id"]):
        return {"group_id": o["chat_id"], "created": False}
    g = chats.create_group(f"이어가기 · {o['title']}", auto=False)
    try:
        ingest.ingest(g["id"], f"📦 {o['title']}", o["text"])
    except HTTPException:
        chats.delete_group(g["id"])  # 실패하면 빈 대화방이 남지 않게 정리
        raise
    outputs.update(oid, chat_id=g["id"])
    return {"group_id": g["id"], "created": True}
