from fastapi import HTTPException
from openai import OpenAI

from app.config import settings


def _client() -> OpenAI:
    if not settings.openai_api_key:
        raise HTTPException(400, "OPENAI_API_KEY가 없습니다. apps/api/.env 파일에 키를 넣어 주세요.")
    return OpenAI(api_key=settings.openai_api_key)


def embed(texts: list[str]) -> list[list[float]]:
    client = _client()
    out: list[list[float]] = []
    for i in range(0, len(texts), 100):
        res = client.embeddings.create(model=settings.embedding_model, input=texts[i : i + 100])
        out.extend(d.embedding for d in res.data)
    return out


def complete(system: str, user: str, model: str | None = None) -> str:
    res = _client().chat.completions.create(
        model=model or settings.chat_model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.2,
    )
    return res.choices[0].message.content or ""
