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


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("통과:", name)
