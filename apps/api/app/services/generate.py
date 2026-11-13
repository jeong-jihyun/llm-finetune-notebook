"""Gemini로 산출물 만들기 / 산출물 이어서 쓰기."""
from fastapi import HTTPException

from app import storage
from app.services import chats, llm, outputs
from app.services.chats import now_str

CONTEXT_CAP = 24000  # 한 번에 Gemini에 보내는 자료 글자 수 (무료 사용량 보호)

SYSTEM = (
    "당신은 사용자가 올린 자료만 근거로 글을 만드는 도우미입니다. "
    "자료에 없는 내용은 지어내지 말고, 한국어로 쉽게 쓰세요. 제목과 목록을 적절히 쓰세요."
)

KINDS = {
    "summary": "위 자료 전체를 처음 보는 사람도 이해하도록 10줄 안쪽으로 요약해 주세요.",
    "faq": "위 자료를 바탕으로 자주 나올 만한 질문 8개와 그 답을 '질문 / 답' 형식으로 써 주세요.",
    "guide": "위 자료로 학습 가이드를 만들어 주세요: 핵심 개념, 용어 정리, 꼭 기억할 점, 복습 질문 5개 순서로.",
    "keypoints": "위 자료의 핵심 포인트를 중요한 순서대로 번호 목록으로 정리해 주세요. 항목마다 한두 문장.",
}


def group_context(gid: str, cap: int = CONTEXT_CAP) -> str:
    """그룹 자료를 자료마다 비슷한 분량으로 잘라 한 덩어리 글로 만든다."""
    docs = storage.list_documents(gid)
    if not docs:
        return ""
    budget = max(cap // len(docs), 500)
    parts = []
    for e in docs:
        chunks, _ = storage.load_chunks(gid, e)
        used, picked = 0, []
        for c in chunks:
            if used + len(c) > budget and picked:
                break
            picked.append(c)
            used += len(c)
        parts.append(f"[자료: {e['title']}]\n" + "\n".join(picked))
    return "\n\n".join(parts)


def make_output(gid: str, kind: str) -> dict:
    if kind not in KINDS:
        raise HTTPException(400, "만들 수 없는 종류입니다")
    group = chats.get_group(gid)
    ctx = group_context(gid)
    if not ctx:
        raise HTTPException(400, "이 대화에 올린 자료가 없어요. 자료를 먼저 올려 주세요.")
    text = llm.complete(SYSTEM, f"[자료]\n{ctx}\n\n[요청]\n{KINDS[kind]}")
    label = outputs.KIND_LABEL[kind]
    return outputs.add(f"{label} · {group['title']}", kind, text, gid)


def save_answer(gid: str, title: str, text: str) -> dict:
    chats.get_group(gid)
    return outputs.add(title or "저장한 답변", "answer", text, gid)


def continue_output(oid: str, instruction: str) -> dict:
    """기존 글 뒤에 Gemini가 이어서 쓴 내용을 붙인다."""
    meta = outputs.get(oid)
    ctx = ""
    if meta.get("group_id") and chats.exists(meta["group_id"]):
        ctx = group_context(meta["group_id"], CONTEXT_CAP // 2)
    ask = instruction.strip() or "앞의 글을 자연스럽게 이어서, 새로운 내용으로 더 자세히 써 주세요."
    user = (
        (f"[참고 자료]\n{ctx}\n\n" if ctx else "")
        + f"[지금까지의 글]\n{meta['text'][-12000:]}\n\n[이어 쓰기 요청]\n{ask}\n"
        + "이미 쓴 내용은 반복하지 말고, 이어질 새 내용만 쓰세요."
    )
    more = llm.complete(SYSTEM, user).strip()
    stamp = now_str()
    new_text = f"{meta['text'].rstrip()}\n\n---\n✍ 이어쓰기 ({stamp})\n\n{more}"
    return outputs.update(oid, text=new_text) | {"text": new_text}
