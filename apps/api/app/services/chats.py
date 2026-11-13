"""대화 그룹 저장소. 그룹 1개 = 대화방 1개 (자료·채팅이 한 묶음).

- chats/index.json  : 그룹 목록 [{id, title, created_at, updated_at, auto}]
- chats/<id>.json   : 그 그룹의 대화 기록 {"messages": [...]}
제목은 첫 질문·답변으로 자동 지어지고(auto=True인 동안), 사용자가 이름을 바꾸면 그대로 유지한다.
저장소 쓰기 횟수를 아끼려고 '최근 활동 시각'은 30분 이상 지났을 때만 목록에 다시 적는다.
"""
import json
import re
import secrets
import time
from datetime import datetime, timedelta, timezone

from fastapi import HTTPException

from app import storage
from app.services import llm

INDEX = "chats/index.json"
_ID_RE = re.compile(r"^g[0-9a-f]{12}$")
MAX_MESSAGES = 200  # 그룹 하나에 남기는 최대 메시지 수 (오래된 것부터 버림)
DEFAULT_TITLE = "새 대화"
TOUCH_MIN = 30
_KST = timezone(timedelta(hours=9))
_FMT = "%Y-%m-%d %H:%M"
_mem: dict = {"groups": None, "t": 0.0}
_TTL = 10


def now_str() -> str:
    return datetime.now(_KST).strftime(_FMT)


def _new_id() -> str:
    return "g" + secrets.token_hex(6)


def _new_group(title: str = DEFAULT_TITLE, auto: bool = True) -> dict:
    t = now_str()
    return {"id": _new_id(), "title": title, "created_at": t, "updated_at": t, "auto": auto}


def _read_index() -> list[dict]:
    if _mem["groups"] is not None and time.time() - _mem["t"] < _TTL:
        return _mem["groups"]
    try:
        groups = json.loads(storage.blob_get(INDEX)).get("groups", [])
    except HTTPException:
        raise
    except Exception:
        groups = []  # 처음 쓰는 경우: 빈 목록으로 시작 (첫 화면에 '새 대화' 버튼이 보임)
    for g in groups:  # 이전 버전 데이터 보정
        g.setdefault("updated_at", g.get("created_at", ""))
        g.setdefault("auto", False)
    _mem.update(groups=groups, t=time.time())
    return groups


def _write_index(groups: list[dict]) -> None:
    storage.blob_put(INDEX, json.dumps({"groups": groups}, ensure_ascii=False).encode("utf-8"))
    _mem.update(groups=groups, t=time.time())


def _check(gid: str) -> dict:
    if not _ID_RE.match(gid):
        raise HTTPException(404, "대화를 찾을 수 없습니다")
    for g in _read_index():
        if g["id"] == gid:
            return g
    raise HTTPException(404, "대화를 찾을 수 없습니다")


def exists(gid: str) -> bool:
    try:
        _check(gid)
        return True
    except HTTPException:
        return False


def get_group(gid: str) -> dict:
    return _check(gid)


def list_groups() -> list[dict]:
    return sorted(_read_index(), key=lambda g: g.get("updated_at", ""), reverse=True)


def create_group(title: str = "", auto: bool | None = None) -> dict:
    title = title.strip()
    g = _new_group(title[:40] or DEFAULT_TITLE, auto=(not title) if auto is None else auto)
    _write_index([g] + list(_read_index()))
    return g


def rename_group(gid: str, title: str) -> dict:
    _check(gid)
    title = title.strip()
    if not title:
        raise HTTPException(400, "이름을 입력해 주세요")
    out = None
    groups = []
    for x in _read_index():
        if x["id"] == gid:
            x = {**x, "title": title[:40], "auto": False}
            out = x
        groups.append(x)
    _write_index(groups)
    return out


def delete_group(gid: str) -> None:
    _check(gid)
    _write_index([x for x in _read_index() if x["id"] != gid])
    for fn in (lambda: storage.blob_del(f"chats/{gid}.json"), lambda: storage.delete_group_docs(gid)):
        try:
            fn()
        except Exception:
            pass


def get_messages(gid: str) -> list[dict]:
    _check(gid)
    try:
        return json.loads(storage.blob_get(f"chats/{gid}.json")).get("messages", [])
    except HTTPException:
        raise
    except Exception:
        return []


def _auto_title(question: str, answer: str) -> str:
    """첫 질문·답변으로 짧은 제목을 짓는다 (Gemini 무료 호출 1회, 실패하면 질문 앞부분)."""
    fallback = question.strip().replace("\n", " ")[:20] or DEFAULT_TITLE
    try:
        t = llm.complete(
            "대화 내용에 어울리는 짧은 한국어 제목을 한 줄만 쓰세요. 15자 안쪽, 따옴표·마침표·설명 없이 제목만.",
            f"[질문]\n{question[:300]}\n\n[답변]\n{answer[:300]}",
        )
        t = re.sub(r"[\"'`“”‘’\n]", "", t).strip(" .")
        return t[:24] or fallback
    except Exception:
        return fallback


def add_exchange(gid: str, question: str, answer: dict) -> None:
    g = _check(gid)
    msgs = get_messages(gid)
    t = now_str()
    msgs.append({"role": "user", "text": question, "t": t})
    msgs.append({"role": "assistant", "text": answer["answer"], "sources": answer.get("sources", []), "t": t})
    msgs = msgs[-MAX_MESSAGES:]
    storage.blob_put(f"chats/{gid}.json", json.dumps({"messages": msgs}, ensure_ascii=False).encode("utf-8"))
    changed = {}
    if g.get("auto") and len(msgs) <= 2:  # 첫 질문이면 제목 자동 짓기
        changed = {"title": _auto_title(question, answer["answer"]), "auto": False, "updated_at": t}
    else:
        try:
            old = datetime.strptime(g.get("updated_at") or "", _FMT).replace(tzinfo=_KST)
            stale = datetime.now(_KST) - old > timedelta(minutes=TOUCH_MIN)
        except ValueError:
            stale = True
        if stale:
            changed = {"updated_at": t}
    if changed:
        _write_index([({**x, **changed} if x["id"] == gid else x) for x in _read_index()])
