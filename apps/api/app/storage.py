"""자료 저장소. Vercel에서는 Vercel Blob(비공개), 내 PC 시험 때는 data/ 폴더를 쓴다.

자료는 대화 그룹마다 따로 둔다.
- docs/<그룹id>/index.json  : 그 그룹의 자료 목록 (목록은 이 파일 하나만 읽어서 만든다 → 저장소 호출 절약)
- docs/<그룹id>/<자료id>.json : 자료 1개 {chunks, dim, emb(base64 float32)}
"""
import base64
import json
import os
import re
import secrets
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from fastapi import HTTPException

from app.config import settings

_GID_RE = re.compile(r"^g[0-9a-f]{12}$")
_ID_RE = re.compile(r"^\d{13}[0-9a-f]{4}$")
_KST = timezone(timedelta(hours=9))
IDX_TTL = 15  # 초. 같은 목록을 자주 읽지 않게 메모리에 잠깐 담아 둔다.
_idx_cache: dict[str, tuple[float, list]] = {}
_doc_cache: dict[str, tuple[list[str], np.ndarray]] = {}


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


# ── 공개 기능 ─────────────────────────────────────────────────────────────
def _check_gid(gid: str) -> None:
    if not _GID_RE.match(gid):
        raise HTTPException(404, "대화를 찾을 수 없습니다")


def _new_id() -> str:
    return f"{int(time.time() * 1000):013d}{secrets.token_hex(2)}"


def _idx_path(gid: str) -> str:
    return f"docs/{gid}/index.json"


def list_documents(gid: str) -> list[dict]:
    _check_gid(gid)
    hit = _idx_cache.get(gid)
    if hit and time.time() - hit[0] < IDX_TTL:
        return hit[1]
    try:
        items = json.loads(blob_get(_idx_path(gid))).get("docs", [])
    except HTTPException:
        raise
    except Exception:
        items = []  # 아직 자료가 없는 그룹
    _idx_cache[gid] = (time.time(), items)
    return items


def _write_index(gid: str, items: list[dict]) -> None:
    blob_put(_idx_path(gid), json.dumps({"docs": items}, ensure_ascii=False).encode("utf-8"))
    _idx_cache[gid] = (time.time(), items)


def add_document(gid: str, title: str, chunks: list[str], embeddings: list[list[float]]) -> dict:
    _check_gid(gid)
    items = list(list_documents(gid))
    doc_id = _new_id()
    mat = np.asarray(embeddings, dtype=np.float32)
    payload = json.dumps(
        {"chunks": chunks, "dim": int(mat.shape[1]), "emb": base64.b64encode(mat.tobytes()).decode()},
        ensure_ascii=False,
    ).encode("utf-8")
    blob_put(f"docs/{gid}/{doc_id}.json", payload)
    meta = {
        "id": doc_id,
        "title": title[:60],
        "chunk_count": len(chunks),
        "created_at": datetime.now(_KST).strftime("%Y-%m-%d %H:%M"),
    }
    _write_index(gid, [meta] + items)
    return meta


def delete_document(gid: str, doc_id: str) -> bool:
    _check_gid(gid)
    if not _ID_RE.match(doc_id):
        return False
    items = list(list_documents(gid))
    if not any(e["id"] == doc_id for e in items):
        return False
    blob_del(f"docs/{gid}/{doc_id}.json")
    _doc_cache.pop(doc_id, None)
    _write_index(gid, [e for e in items if e["id"] != doc_id])
    return True


def delete_group_docs(gid: str) -> None:
    """대화 그룹을 지울 때 그 그룹의 자료도 함께 지운다."""
    for e in list(list_documents(gid)):
        try:
            blob_del(f"docs/{gid}/{e['id']}.json")
        except Exception:
            pass
        _doc_cache.pop(e["id"], None)
    try:
        blob_del(_idx_path(gid))
    except Exception:
        pass
    _idx_cache.pop(gid, None)


def load_chunks(gid: str, entry: dict) -> tuple[list[str], np.ndarray]:
    if entry["id"] not in _doc_cache:
        data = json.loads(blob_get(f"docs/{gid}/{entry['id']}.json"))
        mat = np.frombuffer(base64.b64decode(data["emb"]), dtype=np.float32).reshape(-1, data["dim"])
        _doc_cache[entry["id"]] = (data["chunks"], mat)
    return _doc_cache[entry["id"]]


def search(gid: str, query_embedding: list[float], k: int = 5) -> list[dict]:
    """이 그룹 자료 중 질문과 가장 비슷한 조각 k개 (코사인 유사도)."""
    q = np.asarray(query_embedding, dtype=np.float32)
    qn = np.linalg.norm(q) + 1e-9
    hits: list[dict] = []
    for e in list_documents(gid):
        chunks, mat = load_chunks(gid, e)
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
