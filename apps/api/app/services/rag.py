from app import storage
from app.config import settings
from app.services import llm

SYSTEM = (
    "당신은 사용자가 이 대화에 올린 자료만 근거로 답하는 도우미입니다. "
    "[자료 목록]은 이 대화에 올라온 자료의 처음 부분입니다. '자료를 파악했나', '무슨 자료야', '요약' 같은 질문에는 이 목록을 보고 어떤 자료인지 쉽게 설명하세요. "
    "그 밖의 질문은 [자료]에 답이 없으면 '올려 주신 자료에서 찾을 수 없습니다'라고 답하되, 자료 목록에서 알 수 있는 내용(어떤 자료가 있는지)은 한 줄 덧붙이세요. "
    "답변 문장 끝에 근거가 된 자료 번호를 [1]처럼 붙이세요. 한국어로 쉽게 답하세요."
)


def _overview(gid: str, per_doc: int = 400, max_docs: int = 8) -> str:
    """올린 자료마다 제목과 첫 부분을 보여 줘서 '무슨 자료인지' 묻는 질문에도 답하게 한다."""
    lines = []
    for e in storage.list_documents(gid)[:max_docs]:
        try:
            chunks, _ = storage.load_chunks(gid, e)
            head = (chunks[0] if chunks else "")[:per_doc].replace("\n", " ")
        except Exception:
            head = ""
        lines.append(f"- {e['title']}: {head}")
    return "\n".join(lines)


IMG_SYSTEM_ADD = (
    " 사용자가 이미지(스크린샷 등)를 함께 보냈다면 그 이미지도 근거로 보고, 보이는 내용을 설명하거나 질문에 답하세요."
)


def answer(gid: str, question: str, images: list[dict] | None = None) -> dict:
    if images and not storage.list_documents(gid):
        # 올린 자료는 없고 이미지만 있는 경우: 이미지만 보고 답한다
        text = llm.complete(SYSTEM + IMG_SYSTEM_ADD, f"[질문]\n{question}", images)
        return {"answer": text, "sources": []}
    if not storage.list_documents(gid):
        return {"answer": "이 대화에는 아직 올린 자료가 없습니다. 오른쪽 자료 칸(폰은 '자료' 탭)에서 파일을 올려 주세요.", "sources": []}
    hits = storage.search(gid, llm.embed([question])[0], settings.top_k)
    if not hits:
        return {"answer": "관련 자료를 찾지 못했습니다. 자료를 다시 올려 보세요.", "sources": []}
    context = "\n\n".join(f"[{i + 1}] ({h['title']})\n{h['content']}" for i, h in enumerate(hits))
    text = llm.complete(SYSTEM + (IMG_SYSTEM_ADD if images else ""), f"[자료 목록]\n{_overview(gid)}\n\n[자료]\n{context}\n\n[질문]\n{question}", images)
    sources = [
        {"n": i + 1, "title": h["title"], "snippet": h["content"][:200], "score": round(h["score"], 3)}
        for i, h in enumerate(hits)
    ]
    return {"answer": text, "sources": sources}
