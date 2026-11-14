"""Gemini API (무료 등급) 호출. 모델 이름은 자주 바뀌므로 사용 가능한 모델을 조회해서 고른다."""
import json
import math
import re
import time

import httpx
from fastapi import HTTPException

from app.config import settings
from app.services import keystore

BASE = "https://generativelanguage.googleapis.com/v1beta"
_BAD = ("preview", "tts", "image", "live", "audio", "thinking", "exp", "robotics", "nano", "vision", "computer", "latest")
_cache: dict = {}
_bad_until: dict[str, float] = {}  # 과부하(503)·시간 초과가 난 모델은 잠시 건너뜀
_last_ok: list[str] = []  # 마지막으로 성공한 모델을 먼저 씀
COOLDOWN = 600


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


def _send(method: str, path: str, json: dict | None = None, timeout: float = 40) -> dict:
    try:
        r = httpx.request(method, f"{BASE}/{path}", headers=_headers(), json=json, timeout=timeout)
    except httpx.TimeoutException:
        raise HTTPException(504, "Gemini가 시간 안에 답하지 않았어요. 잠시 뒤 다시 시도해 주세요.")
    except httpx.HTTPError as e:
        raise HTTPException(502, f"Gemini에 연결하지 못했어요 ({type(e).__name__}). 잠시 뒤 다시 시도해 주세요.")
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
    found = pick_chat_models(list_models())
    stable = [n for n in ("gemini-2.5-flash", "gemini-2.5-flash-lite") if n in found]  # 안정판: 빠르고 한도 넉넉
    rest = [n for n in found if n not in stable]
    return (stable + rest)[:5] or ["gemini-flash-latest"]


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


MAX_OUT = 2048  # 답변 길이 상한 (끝없이 길어져 시간 초과되는 것 방지)


def _gen_config(model: str, fast: bool = True) -> dict:
    """새 Gemini(3 계열)는 기본이 '깊이 생각하기'라 느리다 → 생각을 최소로 줄인다."""
    cfg: dict = {"maxOutputTokens": MAX_OUT}
    if not fast:
        return cfg
    if re.search(r"gemini-3", model):
        cfg["thinkingConfig"] = {"thinkingLevel": "low"}  # 3.8은 minimal 불가, low부터 가능
    elif re.search(r"gemini-2\.5", model):
        cfg["thinkingConfig"] = {"thinkingBudget": 0}
    else:
        cfg["temperature"] = 0.2
    return cfg


def _gen(model: str, body: dict, first_byte: float, deadline: float) -> str:
    """스트리밍으로 받는다: 답이 만들어지는 대로 조각이 와서, 오래 조용해서 끊기는 일이 없다."""
    url = f"{BASE}/models/{model}:streamGenerateContent?alt=sse"
    text: list[str] = []
    cut = False
    try:
        with httpx.stream("POST", url, headers=_headers(), json=body,
                          timeout=httpx.Timeout(10, read=first_byte)) as r:
            if r.status_code >= 400:
                r.read()
                _fail(r)
            for line in r.iter_lines():
                if time.monotonic() > deadline:
                    cut = True
                    break
                if not line.startswith("data:"):
                    continue
                try:
                    d = json.loads(line[5:])
                except ValueError:
                    continue
                parts = (d.get("candidates") or [{}])[0].get("content", {}).get("parts", [])
                text += [p["text"] for p in parts if p.get("text") and not p.get("thought")]
    except httpx.TimeoutException:
        if not text:
            raise HTTPException(504, "Gemini가 시간 안에 답하지 않았어요. 잠시 뒤 다시 시도해 주세요.")
        cut = True
    except httpx.HTTPError as e:
        if not text:
            raise HTTPException(502, f"Gemini에 연결하지 못했어요 ({type(e).__name__}). 잠시 뒤 다시 시도해 주세요.")
        cut = True
    out = "".join(text)
    if out and cut:
        out += "\n\n(시간이 걸려 여기까지만 받았어요. 이어서 물어보면 계속 설명해 드려요.)"
    return out


def complete(system: str, user: str) -> str:
    body = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
    }
    last: HTTPException | None = None
    deadline = time.monotonic() + 50  # Vercel 함수 제한(60초) 안에서 끝내기
    models = chat_model()
    now = time.monotonic()
    ready = [m for m in models if _bad_until.get(m, 0) <= now] or models  # 전부 쉬는 중이면 다 시도
    ready.sort(key=lambda m: 0 if m in _last_ok else 1)
    for model in ready:
        if deadline - time.monotonic() < 8:
            break
        t0 = time.monotonic()
        try:
            try:
                text = _gen(model, {**body, "generationConfig": _gen_config(model)}, 20, deadline)
            except HTTPException as e:
                if e.status_code != 502 or "thinking" not in str(e.detail).lower():
                    raise
                # 이 모델이 생각 설정을 거부하면 설정 없이 한 번 더
                text = _gen(model, {**body, "generationConfig": _gen_config(model, False)}, 20, deadline)
        except HTTPException as e:
            print(f"[gemini] {model} 실패 {e.status_code} {time.monotonic() - t0:.1f}초", flush=True)
            if e.status_code == 429:
                raise
            last = e  # 이 모델이 안 되면(과부하·시간 초과 포함) 다음 후보로
            _bad_until[model] = time.monotonic() + COOLDOWN
            continue
        print(f"[gemini] {model} 성공 {time.monotonic() - t0:.1f}초 {len(text)}자", flush=True)
        _last_ok[:] = [model]
        _bad_until.pop(model, None)
        return text or "답변을 만들지 못했습니다."
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
