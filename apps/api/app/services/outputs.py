"""산출물 저장소: Gemini가 만들어 준 글(요약·FAQ·학습 가이드·핵심 정리·저장한 답변)을 한곳에서 관리한다.

- outputs/index.json   : 목록 [{id, title, kind, group_id, created_at, updated_at, chars, chat_id}]
- outputs/<id>.json    : 본문 {"text": "..."}
산출물은 대화를 지워도 남는다 (결과물이니까). chat_id = '대화로 이어가기'로 만든 대화방.
"""
import json
import re
import secrets
import time

from fastapi import HTTPException

from app import storage
from app.services.chats import now_str

INDEX = "outputs/index.json"
_ID_RE = re.compile(r"^o\d{13}[0-9a-f]{4}$")
KIND_LABEL = {"summary": "요약", "faq": "FAQ", "guide": "학습 가이드", "keypoints": "핵심 정리", "answer": "저장한 답변"}
MAX_CHARS = 60000
_mem: dict = {"items": None, "t": 0.0}
_TTL = 10


def _read() -> list[dict]:
    if _mem["items"] is not None and time.time() - _mem["t"] < _TTL:
        return _mem["items"]
    try:
        items = json.loads(storage.blob_get(INDEX)).get("outputs", [])
    except HTTPException:
        raise
    except Exception:
        items = []
    _mem.update(items=items, t=time.time())
    return items


def _write(items: list[dict]) -> None:
    storage.blob_put(INDEX, json.dumps({"outputs": items}, ensure_ascii=False).encode("utf-8"))
    _mem.update(items=items, t=time.time())


def _check(oid: str) -> dict:
    if not _ID_RE.match(oid):
        raise HTTPException(404, "산출물을 찾을 수 없습니다")
    for o in _read():
        if o["id"] == oid:
            return o
    raise HTTPException(404, "산출물을 찾을 수 없습니다")


def list_outputs() -> list[dict]:
    return sorted(_read(), key=lambda o: o.get("updated_at", ""), reverse=True)


def get_text(oid: str) -> str:
    _check(oid)
    try:
        return json.loads(storage.blob_get(f"outputs/{oid}.json")).get("text", "")
    except HTTPException:
        raise
    except Exception:
        return ""


def get(oid: str) -> dict:
    o = _check(oid)
    return {**o, "text": get_text(oid)}


def add(title: str, kind: str, text: str, group_id: str = "") -> dict:
    text = text.strip()
    if not text:
        raise HTTPException(400, "저장할 내용이 없습니다")
    oid = f"o{int(time.time() * 1000):013d}{secrets.token_hex(2)}"
    storage.blob_put(f"outputs/{oid}.json", json.dumps({"text": text[:MAX_CHARS]}, ensure_ascii=False).encode("utf-8"))
    t = now_str()
    meta = {"id": oid, "title": title.strip()[:60] or KIND_LABEL.get(kind, "산출물"), "kind": kind,
            "group_id": group_id, "created_at": t, "updated_at": t, "chars": len(text[:MAX_CHARS]), "chat_id": ""}
    _write([meta] + list(_read()))
    return meta


def update(oid: str, title: str | None = None, text: str | None = None, chat_id: str | None = None) -> dict:
    o = _check(oid)
    new = dict(o)
    if title is not None:
        if not title.strip():
            raise HTTPException(400, "제목을 입력해 주세요")
        new["title"] = title.strip()[:60]
    if chat_id is not None:
        new["chat_id"] = chat_id
    if text is not None:
        text = text.strip()[:MAX_CHARS]
        storage.blob_put(f"outputs/{oid}.json", json.dumps({"text": text}, ensure_ascii=False).encode("utf-8"))
        new["chars"] = len(text)
    if title is not None or text is not None:
        new["updated_at"] = now_str()
    _write([(new if x["id"] == oid else x) for x in _read()])
    return new


def delete(oid: str) -> None:
    _check(oid)
    _write([x for x in _read() if x["id"] != oid])
    try:
        storage.blob_del(f"outputs/{oid}.json")
    except Exception:
        pass
