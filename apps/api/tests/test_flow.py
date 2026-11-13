"""가짜 임베딩/답변으로 업로드→질문→삭제 흐름과 모델 선택, 비밀번호 확인을 점검한다.
실행: cd apps/api && python -m pytest -q   (또는 python tests/test_flow.py)
"""
import hashlib
import io
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["APP_PASSWORD"] = "test-pw"
os.environ.pop("BLOB_READ_WRITE_TOKEN", None)

import numpy as np  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app import storage  # noqa: E402
from app.config import settings  # noqa: E402
from app.services import llm  # noqa: E402

settings.app_password = "test-pw"
settings.data_dir = tempfile.mkdtemp()
H = {"X-App-Password": "test-pw"}


def fake_embed(texts):
    out = []
    for t in texts:
        v = np.zeros(settings.embed_dim)
        for w in t.lower().split():
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % settings.embed_dim] += 1
        out.append((v / (np.linalg.norm(v) or 1)).tolist())
    return out


llm.embed = fake_embed
llm.complete = lambda system, user: "테스트 답변 [1]"
from app.main import app  # noqa: E402

c = TestClient(app)


def test_auth():
    assert c.get("/documents").status_code == 401
    assert c.get("/documents", headers={"X-App-Password": "wrong"}).status_code == 401
    assert c.get("/documents", headers=H).status_code == 200
    assert c.get("/health").json() == {"ok": True}
    assert "자료 노트북" in c.get("/").text


def test_flow():
    r = c.post("/documents", headers=H, files={"file": ("a.txt", "파인튜닝은 모델을 추가로 학습시키는 방법입니다. 라그는 검색해서 답하는 방법입니다.".encode(), "text/plain")})
    assert r.status_code == 200 and r.json()["chunk_count"] == 1
    r = c.post("/documents", headers=H, files={"file": ("b.txt", "한글 cp949 문서 사과 바나나".encode("cp949"), "text/plain")})
    assert r.status_code == 200
    from reportlab.pdfgen import canvas
    b = io.BytesIO(); p = canvas.Canvas(b); p.drawString(72, 720, "retrieval augmented generation pdf test"); p.save()
    assert c.post("/documents", headers=H, files={"file": ("c.pdf", b.getvalue(), "application/pdf")}).status_code == 200
    r = c.post("/documents/text", headers=H, json={"title": "붙여넣기 제목", "text": "붙여넣은 글입니다 고양이 강아지"})
    assert r.status_code == 200
    storage._invalidate()
    docs = c.get("/documents", headers=H).json()
    assert [d["title"] for d in docs][:2] == ["붙여넣기 제목", "c.pdf"] and len(docs) == 4
    r = c.post("/chat", headers=H, json={"question": "라그는 무엇인가요?"}).json()
    assert r["sources"][0]["title"] == "a.txt" and "[1]" in r["answer"]
    assert c.post("/chat", headers=H, json={"question": " "}).status_code == 400
    doc_id = docs[0]["id"]
    assert c.delete(f"/documents/{doc_id}", headers=H).json() == {"ok": True}
    assert c.delete(f"/documents/{doc_id}", headers=H).status_code == 404
    assert c.delete("/documents/..%2F..%2Fetc", headers=H).status_code in (404, 405)
    assert len(c.get("/documents", headers=H).json()) == 3


def test_errors():
    assert c.post("/documents", headers=H, files={"file": ("x.exe", b"zz", "application/octet-stream")}).status_code == 400
    assert c.post("/documents", headers=H, files={"file": ("y.pdf", b"notpdf", "application/pdf")}).status_code == 400
    big = "가나다라 " * 100000
    assert c.post("/documents/text", headers=H, json={"title": "t", "text": big}).status_code == 413
    d = c.get("/diag", headers=H).json()
    assert d["storage"]["ok"] is True and d["storage"]["backend"] == "local-files"
    assert d["gemini"]["ok"] is False  # 키 없음


def test_model_pick():
    models = [
        {"name": "models/gemini-2.5-flash", "supportedGenerationMethods": ["generateContent"]},
        {"name": "models/gemini-3.5-flash", "supportedGenerationMethods": ["generateContent"]},
        {"name": "models/gemini-3.5-flash-preview-05", "supportedGenerationMethods": ["generateContent"]},
        {"name": "models/gemini-3.5-flash-tts", "supportedGenerationMethods": ["generateContent"]},
        {"name": "models/gemini-3.1-flash-lite", "supportedGenerationMethods": ["generateContent"]},
        {"name": "models/gemini-3.5-pro", "supportedGenerationMethods": ["generateContent"]},
        {"name": "models/gemini-embedding-001", "supportedGenerationMethods": ["embedContent", "batchEmbedContents"]},
        {"name": "models/gemini-embedding-2", "supportedGenerationMethods": ["embedContent"]},
        {"name": "models/gemini-embedding-2-preview", "supportedGenerationMethods": ["embedContent"]},
    ]
    assert llm.pick_chat_models(models) == ["gemini-3.5-flash", "gemini-3.1-flash-lite", "gemini-2.5-flash"]
    assert llm.pick_embed_models(models) == ["gemini-embedding-2", "gemini-embedding-001"]


def test_gemini_key_popup_api():
    """화면에서 넣는 Gemini 키: 필요한 비밀번호, 잘못된 모양 거절, 구글 확인 후 저장, 끝 4자리만 노출, 삭제."""
    import httpx
    from app.services import keystore

    keystore.clear_key(); settings.gemini_api_key = ""
    assert c.get("/settings/gemini").status_code == 401
    assert c.get("/settings/gemini", headers=H).json()["has_key"] is False
    try:
        llm._headers(); raise AssertionError("키 없는데 통과함")
    except Exception as e:
        assert "Gemini 키가 아직 없어요" in str(e.detail)
    assert c.post("/settings/gemini", headers=H, json={"key": "짧음"}).status_code == 400

    real_get = httpx.get
    httpx.get = lambda url, **kw: httpx.Response(400, text="API key not valid")
    assert c.post("/settings/gemini", headers=H, json={"key": "AIza" + "x" * 35}).status_code == 400
    assert c.get("/settings/gemini", headers=H).json()["has_key"] is False  # 틀린 키는 저장 안 됨
    httpx.get = lambda url, **kw: httpx.Response(200, json={"models": []})
    r = c.post("/settings/gemini", headers=H, json={"key": " \"AIza" + "y" * 31 + "WXYZ\" "})
    httpx.get = real_get
    assert r.status_code == 200 and r.json()["tail"] == "…WXYZ"
    st = c.get("/settings/gemini", headers=H).json()
    assert st == {"has_key": True, "source": "screen", "tail": "…WXYZ"} and "yyyy" not in str(st)
    assert llm._headers()["x-goog-api-key"].endswith("WXYZ")
    keystore._mem["loaded"] = False  # 다른 서버 인스턴스처럼 저장소에서 다시 읽어도 남아 있어야 함
    assert keystore.get_key().endswith("WXYZ")
    assert c.delete("/settings/gemini", headers=H).json()["source"] == "none"


def test_chat_groups_and_search():
    """채팅 그룹(만들기·이름바꾸기·삭제), 대화 저장, 자료+채팅 통합 검색."""
    assert c.get("/chats").status_code == 401
    assert c.get("/search?q=x").status_code == 401
    groups = c.get("/chats", headers=H).json()
    assert len(groups) >= 1  # 처음엔 기본 그룹이 자동으로 생김
    g = c.post("/chats", headers=H, json={"title": "여행 질문"}).json()
    gid = g["id"]
    assert c.get(f"/chats/{gid}", headers=H).json()["messages"] == []
    assert c.get("/chats/gnotexist0000", headers=H).status_code == 404
    assert c.get("/chats/../x", headers=H).status_code in (404, 405)

    up = c.post("/documents/text", headers=H, json={"title": "제주도 안내", "text": "제주도 여행에서 흑돼지와 감귤을 꼭 먹어보세요. 렌터카는 미리 예약하세요."}).json()
    r = c.post(f"/chats/{gid}/ask", headers=H, json={"question": "제주도에서 뭘 먹을까?"})
    assert r.status_code == 200 and r.json()["answer"]
    msgs = c.get(f"/chats/{gid}", headers=H).json()["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant"] and msgs[0]["text"] == "제주도에서 뭘 먹을까?"
    assert any(x["id"] == gid and x["title"] == "여행 질문" for x in c.get("/chats", headers=H).json())

    # 다른 그룹의 대화는 섞이지 않음
    g2 = c.post("/chats", headers=H, json={"title": "다른 방"}).json()
    assert c.get(f"/chats/{g2['id']}", headers=H).json()["messages"] == []
    c.post(f"/chats/{g2['id']}/ask", headers=H, json={"question": "렌터카 예약은?"})
    assert c.get(f"/chats/{g2['id']}", headers=H).json()["messages"][0]["text"] == "렌터카 예약은?"
    assert len(c.get(f"/chats/{gid}", headers=H).json()["messages"]) == 2

    # 검색: 자료 본문, 자료 제목, 채팅 기록, 여러 단어(AND), 없는 말
    s1 = c.get("/search", headers=H, params={"q": "흑돼지"}).json()
    assert any("흑돼지" in sn for d in s1["docs"] for sn in d["snippets"])
    assert any(h["group_id"] == gid for h in s1["chats"]) is False  # 질문엔 '흑돼지'가 없음
    s2 = c.get("/search", headers=H, params={"q": "제주도 안내"}).json()
    assert any(d["title_hit"] for d in s2["docs"])
    s3 = c.get("/search", headers=H, params={"q": "렌터카"}).json()
    assert any(h["group_id"] == g2["id"] and h["role"] == "user" for h in s3["chats"])
    s4 = c.get("/search", headers=H, params={"q": "제주도 먹을까"}).json()
    assert any(h["group_id"] == gid for h in s4["chats"])
    assert c.get("/search", headers=H, params={"q": "zzzz없는말"}).json() == {"docs": [], "chats": []}
    assert c.get("/search", headers=H, params={"q": "  "}).json() == {"docs": [], "chats": []}

    # 이름 바꾸기, 삭제
    assert c.patch(f"/chats/{gid}", headers=H, json={"title": "제주 여행"}).json()["title"] == "제주 여행"
    assert c.patch(f"/chats/{gid}", headers=H, json={"title": " "}).status_code == 400
    assert c.delete(f"/chats/{gid}", headers=H).status_code == 200
    assert all(x["id"] != gid for x in c.get("/chats", headers=H).json())
    assert c.get(f"/chats/{gid}", headers=H).status_code == 404
    for x in c.get("/chats", headers=H).json():  # 전부 지워도 빈 기본 그룹이 남음
        c.delete(f"/chats/{x['id']}", headers=H)
    left = c.get("/chats", headers=H).json()
    assert len(left) == 1 and left[0]["title"] == "기본 대화"
    c.delete(f"/documents/{up['id']}", headers=H)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("통과:", name)
