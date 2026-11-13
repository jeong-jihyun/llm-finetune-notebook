"""채팅 그룹(대화방) 저장소.

- chats/index.json      : 그룹 목록 [{id, title, created_at}]
- chats/<id>.json       : 그룹 하나의 대화 기록 {"messages": [...]}
저장소 호출을 아끼려고 질문+답변 한 쌍을 한 번에 저장합니다 (저장 1회).
"""
import json
import re
import secrets
import time
from datetime import datetime, timedelta, timezone

from fastapi import HTTPException

from app import storage

INDEX = "chats/index.json"
_ID_RE = re.compile(r"^g[0-9a-f]{12}$")
MAX_MESSAGES = 200  # 그룹 하나에 남기는 최대 메시지 수 (오래된 것부터 버림)
DEFAULT_TITLE = "새 대화"
_KST = timezone(timedelta(hours=9))
_mem: dict = {"groups": None, "t": 0.0}
_TTL = 10


def _now() -> str:
    return datetime.now(_KST).strftime("%Y-%m-%d %H:%M")


def _read_index() -> list[dict]:
    if _mem["groups"] is not None and time.time() - _mem["t"] < _TTL:
        return _mem["groups"]
    try:
        groups = json.loads(storage.blob_get(INDEX)).get("groups", [])
    except Exception:
        groups = None
    if groups is None:  # 처음 쓰는 경우: 기본 그룹 하나 만들기
        groups = [{"id": _new_id(), "title": "기본 대화", "created_at": _now()}]
        _write_index(groups)
    _mem.update(groups=groups, t=time.time())
    return groups


def _write_index(groups: list[dict]) -> None:
    storage.blob_put(INDEX, json.dumps({"groups": groups}, ensure_ascii=False).encode("utf-8"))
    _mem.update(groups=groups, t=time.time())


def _new_id() -> str:
    return "g" + secrets.token_hex(6)


def _check(gid: str) -> dict:
    if not _ID_RE.match(gid):
        raise HTTPException(404, "대화 그룹을 찾을 수 없습니다")
    for g in _read_index():
        if g["id"] == gid:
            return g
    raise HTTPException(404, "대화 그룹을 찾을 수 없습니다")


def list_groups() -> list[dict]:
    return list(_read_index())


def create_group(title: str) -> dict:
    groups = list(_read_index())
    g = {"id": _new_id(), "title": (title.strip() or DEFAULT_TITLE)[:40], "created_at": _now()}
    groups.insert(0, g)
    _write_index(groups)
    return g


def rename_group(gid: str, title: str) -> dict:
    g = _check(gid)
    title = title.strip()
    if not title:
        raise HTTPException(400, "이름을 입력해 주세요")
    groups = [({**x, "title": title[:40]} if x["id"] == gid else x) for x in _read_index()]
    _write_index(groups)
    return {**g, "title": title[:40]}


def delete_group(gid: str) -> None:
    _check(gid)
    groups = [x for x in _read_index() if x["id"] != gid]
    if not groups:  # 마지막 그룹을 지워도 화면이 비지 않게 빈 기본 그룹을 새로 만듦
        groups = [{"id": _new_id(), "title": "기본 대화", "created_at": _now()}]
    _write_index(groups)
    try:
        storage.blob_del(f"chats/{gid}.json")
    except Exception:
        pass


def get_messages(gid: str) -> list[dict]:
    _check(gid)
    try:
        return json.loads(storage.blob_get(f"chats/{gid}.json")).get("messages", [])
    except Exception:
        return []


def add_exchange(gid: str, question: str, answer: dict) -> None:
    g = _check(gid)
    msgs = get_messages(gid)
    t = _now()
    msgs.append({"role": "user", "text": question, "t": t})
    msgs.append({"role": "assistant", "text": answer["answer"], "sources": answer.get("sources", []), "t": t})
    msgs = msgs[-MAX_MESSAGES:]
    storage.blob_put(f"chats/{gid}.json", json.dumps({"messages": msgs}, ensure_ascii=False).encode("utf-8"))
    if g["title"] in (DEFAULT_TITLE, "기본 대화") and len(msgs) <= 2:  # 첫 질문으로 그룹 이름 자동 짓기
        rename_group(gid, question.strip().replace("\n", " ")[:20])
