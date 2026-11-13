"""통합 검색: 자료(제목+본문)와 채팅 기록을 글자 그대로 찾는다 (AI 호출 없음, 무료 한도에 안 잡힘)."""
from app import storage
from app.services import chats

MAX_DOCS, MAX_SNIPS, MAX_CHAT_HITS = 20, 2, 30


def _terms(q: str) -> list[str]:
    return [t for t in q.lower().split() if t]


def _snippet(text: str, terms: list[str], before: int = 40, after: int = 110) -> str:
    low = text.lower()
    pos = min((low.find(t) for t in terms if low.find(t) >= 0), default=0)
    s = max(0, pos - before)
    out = text[s : pos + after].replace("\n", " ").strip()
    return ("…" if s > 0 else "") + out + ("…" if pos + after < len(text) else "")


def _has_all(text: str, terms: list[str]) -> bool:
    low = text.lower()
    return all(t in low for t in terms)


def search_docs(terms: list[str]) -> list[dict]:
    out = []
    for e in storage._entries():
        title_hit = _has_all(e["title"], terms)
        snippets = []
        try:
            chunks, _ = storage._load(e)
        except Exception:
            chunks = []
        for c in chunks:
            if _has_all(c, terms):
                snippets.append(_snippet(c, terms))
                if len(snippets) >= MAX_SNIPS:
                    break
        if title_hit or snippets:
            out.append({"id": e["id"], "title": e["title"], "title_hit": title_hit, "snippets": snippets})
        if len(out) >= MAX_DOCS:
            break
    return out


def search_chats(terms: list[str]) -> list[dict]:
    out = []
    for g in chats.list_groups():
        for i, m in enumerate(chats.get_messages(g["id"])):
            if _has_all(m["text"], terms):
                out.append(
                    {"group_id": g["id"], "group_title": g["title"], "index": i, "role": m["role"],
                     "snippet": _snippet(m["text"], terms), "t": m.get("t", "")}
                )
                if len(out) >= MAX_CHAT_HITS:
                    return out
    return out


def search(q: str) -> dict:
    terms = _terms(q)
    if not terms:
        return {"docs": [], "chats": []}
    return {"docs": search_docs(terms), "chats": search_chats(terms)}
