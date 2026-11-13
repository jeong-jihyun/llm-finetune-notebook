"""자료 저장소. Vercel에서는 Vercel Blob(비공개), 내 PC 시험 때는 data/ 폴더를 쓴다.

자료 1개 = 파일 1개 (docs/<id>__<조각수>__<제목base64>.json). 목록은 파일 이름만 읽어서 만든다.
"""
import base64
import json
import os
import re
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from fastapi import HTTPException

from app.config import settings

PREFIX = "docs/"
_ID_RE = re.compile(r"^\d{13}[0-9a-f]{4}$")
_NAME_RE = re.compile(r"^docs/(\d{13}[0-9a-f]{4})__(\d+)__([A-Za-z0-9_-]*)\.json$")

_list_cache: dict = {"t": 0.0, "items": None}
_doc_cache: dict[str, tuple[list[str], np.ndarray]] = {}
LIST_TTL = 20  # 초. 목록 조회 횟수를 줄여 무료 사용량을 아낀다.


# ── 저장 방식 4가지 기본 동작 (시험 때 바꿔 끼울 수 있게 분리) ─────────────
def _use_blob() -> bool:
    return bool(os.environ.get("BLOB_READ_WRITE_TOKEN"))


def _guard() -> None:
    if not _use_blob() and os.environ.get("VERCEL"):
        raise HTTPException(500, "저장소(Vercel Blob)가 프로젝트에 연결되지 않았습니다")


def blob_put(path: str, data: bytes) -> None:
    _guard()
    if _use_blob():
        from vercel.blob.sync import put_blob

        put_blob(path, data, access="private", allow_overwrite=True, content_type="application/json")
    else:
        p = Path(settings.data_dir) / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)


def blob_get(path: str) -> bytes:
    _guard()
    if _use_blob():
        from vercel.blob.sync import get_blob

        return get_blob(path, access="private").body
    return (Path(settings.data_dir) / path).read_bytes()


def blob_list(prefix: str) -> list[str]:
    _guard()
    if _use_blob():
        from vercel.blob.sync import query_blobs

        return [b.pathname for b in query_blobs(prefix=prefix)]
    d = Path(settings.data_dir) / prefix
    return [prefix + f.name for f in d.glob("*")] if d.exists() else []


def blob_del(path: str) -> None:
    _guard()
    if _use_blob():
        from vercel.blob.sync import delete_blob

        delete_blob(path)
    else:
        (Path(settings.data_dir) / path).unlink(missing_ok=True)


# ── 이름 규칙 ─────────────────────────────────────────────────────────────
def _enc_title(title: str) -> str:
    return base64.urlsafe_b64encode(title[:60].encode("utf-8")).decode().rstrip("=")


def _dec_title(b64: str) -> str:
    try:
        return base64.urlsafe_b64decode(b64 + "=" * (-len(b64) % 4)).decode("utf-8")
    except Exception:
        return "(제목 없음)"


def _new_id() -> str:
    return f"{int(time.time() * 1000):013d}{secrets.token_hex(2)}"


def _invalidate() -> None:
    _list_cache["items"] = None


# ── 공개 기능 ─────────────────────────────────────────────────────────────
def add_document(title: str, chunks: list[str], embeddings: list[list[float]]) -> str:
    doc_id = _new_id()
    mat = np.asarray(embeddings, dtype=np.float32)
    payload = json.dumps(
        {"chunks": chunks, "dim": int(mat.shape[1]), "emb": base64.b64encode(mat.tobytes()).decode()},
        ensure_ascii=False,
    ).encode("utf-8")
    blob_put(f"{PREFIX}{doc_id}__{len(chunks)}__{_enc_title(title)}.json", payload)
    _invalidate()
    return doc_id


def _entries() -> list[dict]:
    now = time.time()
    if _list_cache["items"] is not None and now - _list_cache["t"] < LIST_TTL:
        return _list_cache["items"]
    items = []
    for path in blob_list(PREFIX):
        m = _NAME_RE.match(path)
        if not m:
            continue
        doc_id, n, b64 = m.group(1), int(m.group(2)), m.group(3)
        items.append(
            {
                "id": doc_id,
                "title": _dec_title(b64),
                "chunk_count": n,
                "created_at": datetime.fromtimestamp(int(doc_id[:13]) / 1000, timezone.utc).strftime(
                    "%Y-%m-%d %H:%M"
                ),
                "_path": path,
            }
        )
    items.sort(key=lambda x: x["id"], reverse=True)
    _list_cache.update(t=now, items=items)
    return items


def list_documents() -> list[dict]:
    return [{k: v for k, v in e.items() if k != "_path"} for e in _entries()]


def delete_document(doc_id: str) -> bool:
    if not _ID_RE.match(doc_id):
        return False
    _invalidate()
    for e in _entries():
        if e["id"] == doc_id:
            blob_del(e["_path"])
            _doc_cache.pop(doc_id, None)
            _invalidate()
            return True
    return False


def _load(entry: dict) -> tuple[list[str], np.ndarray]:
    if entry["id"] not in _doc_cache:
        data = json.loads(blob_get(entry["_path"]))
        mat = np.frombuffer(base64.b64decode(data["emb"]), dtype=np.float32).reshape(-1, data["dim"])
        _doc_cache[entry["id"]] = (data["chunks"], mat)
    return _doc_cache[entry["id"]]


def search(query_embedding: list[float], k: int = 5) -> list[dict]:
    """질문과 가장 비슷한 조각 k개 (코사인 유사도)."""
    q = np.asarray(query_embedding, dtype=np.float32)
    qn = np.linalg.norm(q) + 1e-9
    hits: list[dict] = []
    for e in _entries():
        chunks, mat = _load(e)
        if mat.shape[1] != q.shape[0]:
            continue  # 임베딩 모델이 바뀐 옛 자료는 건너뜀
        sims = (mat @ q) / (np.linalg.norm(mat, axis=1) * qn + 1e-9)
        for i in np.argsort(-sims)[:k]:
            hits.append({"title": e["title"], "content": chunks[int(i)], "score": float(sims[int(i)])})
    hits.sort(key=lambda h: -h["score"])
    return hits[:k]


def check() -> dict:
    """저장소 읽기/쓰기/삭제가 되는지 확인 (시스템 점검 버튼용)."""
    path = f"check/{secrets.token_hex(4)}.json"
    try:
        blob_put(path, b'{"ok":true}')
        ok = json.loads(blob_get(path)).get("ok") is True
        blob_del(path)
        return {"backend": "vercel-blob" if _use_blob() else "local-files", "ok": ok}
    except Exception as e:  # noqa: BLE001
        return {"backend": "vercel-blob" if _use_blob() else "local-files", "ok": False, "error": str(e)[:200]}
