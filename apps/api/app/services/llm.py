"""Gemini API (무료 등급) 호출. 모델 이름은 자주 바뀌므로 사용 가능한 모델을 조회해서 고른다."""
import math
import re

import httpx
from fastapi import HTTPException

from app.config import settings
from app.services import keystore

BASE = "https://generativelanguage.googleapis.com/v1beta"
_BAD = ("preview", "tts", "image", "live", "audio", "thinking", "exp", "robotics", "nano", "vision", "computer", "latest")
_cache: dict = {}


def _headers() -> dict:
    key = keystore.get_key()
    if not key:
        raise HTTPException(400, "Gemini 키가 아직 없어요. 화면의 'Gemini 키 넣기'에서 무료 키를 넣어 주세요.")
    return {"x-goog-api-key": key, "Content-Type": "application/json"}


def _fail(r: httpx.Response) -> None:
    if r.status_code == 429:
        raise HTTPException(429, "Gemini 무료 사용 한도에 도달했어요. 잠시 뒤(길면 내일) 다시 시도해 주세요.")
    if r.status_code in (400, 401, 403) and "API key" in r.text:
        raise HTTPException(400, "Gemini API 키가 올바르지 않습니다. 키를 다시 확인해 주세요.")
    raise HTTPException(502, f"Gemini 오류 {r.status_code}: {r.text[:200]}")


def _send(method: str, path: str, json: dict | None = None) -> dict:
    r = httpx.request(method, f"{BASE}/{path}", headers=_headers(), json=json, timeout=55)
    if r.status_code >= 400:
        _fail(r)
    return r.json()


def list_models() -> list[dict]:
    if "models" not in _cache:
        out, token = [], ""
        while True:
            data = _send("GET", f"models?pageSize=200{('&pageToken=' + token) if token else ''}")
            out += data.get("models", [])
            token = data.get("nextPageToken", "")
            if not token:
                break
        _cache["models"] = out
    return _cache["models"]


def _ver(name: str) -> float:
    m = re.search(r"-(\d+(?:\.\d+)?)", name)
    return float(m.group(1)) if m else 0.0


def pick_chat_models(models: list[dict]) -> list[str]:
    """generateContent 가능한 안정판 flash 계열을 최신순으로. 같은 버전이면 lite(무료 한도 넉넉)를 먼저."""
    names = [
        m["name"].removeprefix("models/")
        for m in models
        if "generateContent" in m.get("supportedGenerationMethods", [])
        and m["name"].removeprefix("models/").startswith("gemini-")
        and "flash" in m["name"]
        and not any(b in m["name"] for b in _BAD)
    ]
    return sorted(names, key=lambda n: (-_ver(n), 0 if "lite" in n else 1))


def pick_embed_models(models: list[dict]) -> list[str]:
    names = [
        m["name"].removeprefix("models/")
        for m in models
        if any(x in m.get("supportedGenerationMethods", []) for x in ("embedContent", "batchEmbedContents"))
        and "embedding" in m["name"]
        and not any(b in m["name"] for b in ("preview", "exp"))
    ]
    return sorted(names, key=lambda n: -_ver(n))


def chat_model() -> list[str]:
    if settings.gemini_chat_model:
        return [settings.gemini_chat_model]
    return pick_chat_models(list_models())[:3] or ["gemini-flash-latest"]


def embed_model() -> str:
    if settings.gemini_embed_model:
        return settings.gemini_embed_model
    found = pick_embed_models(list_models())
    if not found:
        raise HTTPException(502, "사용 가능한 Gemini 임베딩 모델을 찾지 못했습니다")
    return found[0]


def _fit(vec: list[float]) -> list[float]:
    """차원을 설정값으로 맞추고 길이를 1로 정규화."""
    v = vec[: settings.embed_dim]
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def embed(texts: list[str]) -> list[list[float]]:
    model = embed_model()
    out: list[list[float]] = []
    for i in range(0, len(texts), 100):
        batch = texts[i : i + 100]
        body = {
            "requests": [
                {
                    "model": f"models/{model}",
                    "content": {"parts": [{"text": t}]},
                    "outputDimensionality": settings.embed_dim,
                }
                for t in batch
            ]
        }
        data = _send("POST", f"models/{model}:batchEmbedContents", body)
        out.extend(_fit(e["values"]) for e in data["embeddings"])
    return out


def complete(system: str, user: str) -> str:
    body = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {"temperature": 0.2},
    }
    last: HTTPException | None = None
    for model in chat_model():
        try:
            data = _send("POST", f"models/{model}:generateContent", body)
        except HTTPException as e:
            if e.status_code == 429:
                raise
            last = e  # 이 모델이 안 되면 다음 후보로
            continue
        parts = (data.get("candidates") or [{}])[0].get("content", {}).get("parts", [])
        return "".join(p.get("text", "") for p in parts) or "답변을 만들지 못했습니다."
    raise last or HTTPException(502, "Gemini 답변 생성에 실패했습니다")


def check() -> dict:
    """Gemini 키와 모델 선택이 되는지 확인 (모델 목록 조회는 사용량에 안 잡힘)."""
    try:
        models = list_models()
        return {
            "ok": True,
            "chat_models": chat_model(),
            "embed_model": embed_model(),
            "model_count": len(models),
        }
    except HTTPException as e:
        return {"ok": False, "error": e.detail}
