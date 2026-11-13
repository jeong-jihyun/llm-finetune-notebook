"""검색: 글자 그대로 찾는다 (AI 호출 없음 → Gemini 무료 한도에 안 잡힘).

- scope=home    : 대화 제목, 채팅 기록, 대화별 자료(제목+본문)
- scope=outputs : 산출물(제목+본문)
"""
from app import storage
from app.services import chats, outputs

MAX_DOCS, MAX_SNIPS, MAX_CHAT_HITS, MAX_OUTPUTS = 20, 2, 30, 100


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


def search_home(terms: list[str]) -> dict:
    groups, chat_hits, doc_hits = [], [], []
    for g in chats.list_groups():
        if _has_all(g["title"], terms):
            groups.append({"id": g["id"], "title": g["title"], "updated_at": g.get("updated_at", "")})
        if len(chat_hits) < MAX_CHAT_HITS:
            for i, m in enumerate(chats.get_messages(g["id"])):
                if _has_all(m["text"], terms):
                    chat_hits.append({"group_id": g["id"], "group_title": g["title"], "index": i, "role": m["role"],
                                      "snippet": _snippet(m["text"], terms), "t": m.get("t", "")})
                    if len(chat_hits) >= MAX_CHAT_HITS:
                        break
        for e in storage.list_documents(g["id"]):
            if len(doc_hits) >= MAX_DOCS:
                break
            title_hit = _has_all(e["title"], terms)
            snippets = []
            try:
                chunks, _ = storage.load_chunks(g["id"], e)
            except Exception:
                chunks = []
            for c in chunks:
                if _has_all(c, terms):
                    snippets.append(_snippet(c, terms))
                    if len(snippets) >= MAX_SNIPS:
                        break
            if title_hit or snippets:
                doc_hits.append({"group_id": g["id"], "group_title": g["title"], "id": e["id"],
                                 "title": e["title"], "snippets": snippets})
    return {"groups": groups, "chats": chat_hits, "docs": doc_hits}


def search_outputs(terms: list[str]) -> dict:
    out = []
    for o in outputs.list_outputs()[:MAX_OUTPUTS]:
        title_hit = _has_all(o["title"], terms)
        text = outputs.get_text(o["id"])
        body_hit = _has_all(text, terms)
        if title_hit or body_hit:
            out.append({**o, "snippet": _snippet(text, terms) if body_hit else ""})
    return {"outputs": out}


def search(q: str, scope: str = "home") -> dict:
    terms = _terms(q)
    if scope == "outputs":
        return search_outputs(terms) if terms else {"outputs": []}
    return search_home(terms) if terms else {"groups": [], "chats": [], "docs": []}
