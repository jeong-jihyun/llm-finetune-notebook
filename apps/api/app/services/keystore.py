"""화면에서 넣은 Gemini 키 보관소.

- 키는 저장소(Vercel Blob 비공개 / 로컬은 data/)의 settings/gemini_key.json 에 둡니다.
- 화면에서 넣은 키가 있으면 그것을 쓰고, 없으면 환경변수 GEMINI_API_KEY 를 씁니다.
- 읽기는 60초 동안 메모리에 담아 두어 저장소 호출 횟수를 아낍니다.
"""
import json
import time

from app import storage
from app.config import settings

PATH = "settings/gemini_key.json"
_TTL = 60
_mem: dict = {"key": None, "t": 0.0, "loaded": False}


def _load() -> str | None:
    if _mem["loaded"] and time.time() - _mem["t"] < _TTL:
        return _mem["key"]
    key = None
    try:
        key = json.loads(storage.blob_get(PATH)).get("key") or None
    except Exception:
        key = None  # 파일이 없거나 저장소 문제 → 환경변수로 대체
    _mem.update(key=key, t=time.time(), loaded=True)
    return key


def get_key() -> str:
    return _load() or settings.gemini_api_key or ""


def source() -> str:
    if _load():
        return "screen"
    return "env" if settings.gemini_api_key else "none"


def masked() -> str:
    k = get_key()
    return ("…" + k[-4:]) if len(k) >= 8 else ""


def set_key(key: str) -> None:
    storage.blob_put(PATH, json.dumps({"key": key}).encode("utf-8"))
    _mem.update(key=key, t=time.time(), loaded=True)


def clear_key() -> None:
    try:
        storage.blob_del(PATH)
    except Exception:
        pass
    _mem.update(key=None, t=time.time(), loaded=True)
