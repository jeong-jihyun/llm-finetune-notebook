from app import storage
from app.config import settings
from app.services import llm

SYSTEM = (
    "당신은 사용자가 올린 자료만 근거로 답하는 도우미입니다. "
    "아래 [자료]에 답이 없으면 '올려 주신 자료에서 찾을 수 없습니다'라고 답하세요. "
    "답변 문장 끝에 근거가 된 자료 번호를 [1]처럼 붙이세요. 한국어로 쉽게 답하세요."
)


def answer(question: str) -> dict:
    if not storage.list_documents():
        return {"answer": "아직 올린 자료가 없습니다. 먼저 자료함에서 파일을 올려 주세요.", "sources": []}
    hits = storage.search(llm.embed([question])[0], settings.top_k)
    if not hits:
        return {"answer": "관련 자료를 찾지 못했습니다. 자료를 다시 올려 보세요.", "sources": []}
    context = "\n\n".join(f"[{i + 1}] ({h['title']})\n{h['content']}" for i, h in enumerate(hits))
    text = llm.complete(SYSTEM, f"[자료]\n{context}\n\n[질문]\n{question}")
    sources = [
        {"n": i + 1, "title": h["title"], "snippet": h["content"][:200], "score": round(h["score"], 3)}
        for i, h in enumerate(hits)
    ]
    return {"answer": text, "sources": sources}
