import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.auth import require_password
from app.services import keystore, llm

router = APIRouter(prefix="/settings", tags=["settings"], dependencies=[Depends(require_password)])


class KeyIn(BaseModel):
    key: str


@router.get("/gemini")
def status():
    """키가 들어 있는지(끝 4자리만) 알려 줍니다. 키 전체는 절대 돌려주지 않아요."""
    return {"has_key": bool(keystore.get_key()), "source": keystore.source(), "tail": keystore.masked()}


@router.post("/gemini")
def save(body: KeyIn):
    key = body.key.strip().strip('"').strip("'")
    if len(key) < 20 or any(c.isspace() for c in key):
        raise HTTPException(400, "키 모양이 아니에요. AI Studio에서 복사한 키 전체(AIza…)를 붙여 넣어 주세요.")
    # 저장하기 전에 구글에 한 번 물어서 진짜 키인지 확인 (모델 목록 조회는 사용량에 안 잡혀요)
    try:
        r = httpx.get(f"{llm.BASE}/models?pageSize=1", headers={"x-goog-api-key": key}, timeout=20)
    except httpx.HTTPError:
        raise HTTPException(502, "구글에 연결하지 못했어요. 잠시 뒤 다시 시도해 주세요.")
    if r.status_code in (400, 401, 403):
        raise HTTPException(400, "구글이 이 키를 받아 주지 않아요. 키를 다시 복사해 보세요.")
    if r.status_code >= 400 and r.status_code != 429:
        raise HTTPException(502, f"구글 확인 오류 {r.status_code}")
    keystore.set_key(key)
    llm._cache.clear()  # 이전 키로 담아 둔 모델 목록 버리기
    return {"ok": True, "tail": keystore.masked()}


@router.delete("/gemini")
def remove():
    keystore.clear_key()
    llm._cache.clear()
    return {"ok": True, "source": keystore.source()}
