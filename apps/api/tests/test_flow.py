"""가짜 임베딩/답변으로 대화 그룹(자료+채팅), 자동 제목, 산출물, 검색, 이어가기를 점검한다.
실행: cd apps/api && python tests/test_flow.py
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
from app.services import chats, llm  # noqa: E402

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


def fake_complete(system, user):
    if "글을 만드는" in system:  # 산출물 만들기 / 이어 쓰기
        return "새로 이어진 내용입니다" if "이어 쓰기 요청" in user else "# 요약\n흑돼지와 감귤 정리"
    if "제목" in system:  # 대화 제목 자동 짓기
        return "\"제주 여행 먹거리\""
    return "테스트 답변 [1]"


REAL_COMPLETE = llm.complete  # 실제 함수(시간 초과 테스트용)
llm.embed = fake_embed
llm.complete = fake_complete
from app.main import app  # noqa: E402

c = TestClient(app)


def mk(title=""):
    return c.post("/chats", headers=H, json={"title": title}).json()


def up_text(gid, title, text):
    return c.post(f"/chats/{gid}/docs/text", headers=H, json={"title": title, "text": text})


def test_auth():
    for path in ("/chats", "/outputs", "/search?q=x", "/settings/gemini"):
        assert c.get(path).status_code == 401, path
        assert c.get(path, headers={"X-App-Password": "wrong"}).status_code == 401, path
    assert c.get("/health").json() == {"ok": True}
    assert "자료 노트북" in c.get("/").text
    assert c.get("/documents", headers=H).status_code == 404  # 옛 주소는 없어짐


def test_groups_auto_title_and_resume():
    """첫 화면 목록: 새 대화 → 첫 질문으로 제목 자동 → 다시 열어 이어서 대화."""
    assert c.get("/chats", headers=H).json() == []
    g = mk()
    assert g["title"] == "새 대화" and g["auto"] is True
    gid = g["id"]
    r = c.post(f"/chats/{gid}/ask", headers=H, json={"question": "제주도에서 뭘 먹을까?"})
    assert r.status_code == 200 and "자료가 없습니다" in r.json()["answer"]
    now = c.get("/chats", headers=H).json()[0]
    assert now["title"] == "제주 여행 먹거리" and now["auto"] is False  # 따옴표 제거된 자동 제목
    c.post(f"/chats/{gid}/ask", headers=H, json={"question": "렌터카는?"})
    assert c.get("/chats", headers=H).json()[0]["title"] == "제주 여행 먹거리"  # 두 번째 질문은 제목을 안 바꿈
    opened = c.get(f"/chats/{gid}", headers=H).json()  # 목록에서 선택해 다시 열기
    assert opened["title"] == "제주 여행 먹거리" and [m["role"] for m in opened["messages"]] == ["user", "assistant"] * 2
    assert c.post(f"/chats/{gid}/ask", headers=H, json={"question": " "}).status_code == 400
    # 이름을 직접 바꾸면 자동 제목이 덮어쓰지 않음, 빈 이름은 거절
    assert c.patch(f"/chats/{gid}", headers=H, json={"title": "내 이름"}).json()["title"] == "내 이름"
    assert c.patch(f"/chats/{gid}", headers=H, json={"title": " "}).status_code == 400
    # 최근 활동 순서
    g2 = mk("직접 지은 이름")
    assert g2["auto"] is False
    for x in chats._read_index():
        x["updated_at"] = "2026-01-01 00:00" if x["id"] == g2["id"] else "2026-02-01 00:00"
    assert [x["id"] for x in c.get("/chats", headers=H).json()] == [gid, g2["id"]]
    assert c.get("/chats/gnotexist0000", headers=H).status_code == 404
    assert c.get("/chats/..%2Fx", headers=H).status_code in (404, 405)
    for x in c.get("/chats", headers=H).json():
        c.delete(f"/chats/{x['id']}", headers=H)
    assert c.get("/chats", headers=H).json() == []


def test_docs_are_per_group():
    a, b = mk("A방")["id"], mk("B방")["id"]
    r = c.post(f"/chats/{a}/docs", headers=H, files={"file": ("a.txt", "파인튜닝은 모델을 추가로 학습시키는 방법입니다. 라그는 검색해서 답하는 방법입니다.".encode(), "text/plain")})
    assert r.status_code == 200 and r.json()["chunk_count"] == 1
    assert c.post(f"/chats/{a}/docs", headers=H, files={"file": ("b.txt", "한글 cp949 문서 사과 바나나".encode("cp949"), "text/plain")}).status_code == 200
    from reportlab.pdfgen import canvas
    buf = io.BytesIO(); p = canvas.Canvas(buf); p.drawString(72, 720, "retrieval augmented generation pdf test"); p.save()
    assert c.post(f"/chats/{a}/docs", headers=H, files={"file": ("c.pdf", buf.getvalue(), "application/pdf")}).status_code == 200
    assert up_text(a, "붙여넣기 제목", "붙여넣은 글입니다 고양이 강아지").status_code == 200
    docs = c.get(f"/chats/{a}/docs", headers=H).json()
    assert [d["title"] for d in docs] == ["붙여넣기 제목", "c.pdf", "b.txt", "a.txt"]
    assert c.get(f"/chats/{b}/docs", headers=H).json() == []  # 다른 대화에는 안 보임
    ra = c.post(f"/chats/{a}/ask", headers=H, json={"question": "라그는 무엇인가요?"}).json()
    assert ra["sources"][0]["title"] == "a.txt" and "[1]" in ra["answer"]
    rb = c.post(f"/chats/{b}/ask", headers=H, json={"question": "라그는 무엇인가요?"}).json()
    assert rb["sources"] == [] and "자료가 없습니다" in rb["answer"]  # B방은 A방 자료로 답하지 않음
    did = docs[0]["id"]
    assert c.delete(f"/chats/{a}/docs/{did}", headers=H).json() == {"ok": True}
    assert c.delete(f"/chats/{a}/docs/{did}", headers=H).status_code == 404
    assert c.delete(f"/chats/{a}/docs/..%2Fx", headers=H).status_code in (404, 405)
    assert len(c.get(f"/chats/{a}/docs", headers=H).json()) == 3
    # 대화를 지우면 그 대화의 자료 파일도 같이 지워짐
    folder = os.path.join(settings.data_dir, "docs", a)
    assert os.path.isdir(folder) and os.listdir(folder)
    c.delete(f"/chats/{a}", headers=H); c.delete(f"/chats/{b}", headers=H)
    assert not os.path.exists(folder) or not os.listdir(folder)


def test_errors():
    g = mk()["id"]
    assert c.post(f"/chats/{g}/docs", headers=H, files={"file": ("x.exe", b"zz", "application/octet-stream")}).status_code == 400
    assert c.post(f"/chats/{g}/docs", headers=H, files={"file": ("y.pdf", b"notpdf", "application/pdf")}).status_code == 400
    assert up_text(g, "t", "가나다라 " * 100000).status_code == 413
    assert c.post("/chats/gnotexist0000/docs/text", headers=H, json={"title": "t", "text": "x"}).status_code == 404
    d = c.get("/diag", headers=H).json()
    assert d["storage"]["ok"] is True and d["storage"]["backend"] == "local-files"
    assert d["gemini"]["ok"] is False  # 키 없음
    c.delete(f"/chats/{g}", headers=H)


def test_outputs_manage_search_continue():
    """산출물 탭: 만들기·저장·보기·이름바꾸기·검색·이어서 쓰기·대화로 이어가기·삭제."""
    g = mk("제주 자료방")["id"]
    assert c.post(f"/chats/{g}/generate", headers=H, json={"kind": "summary"}).status_code == 400  # 자료 없이 못 만듦
    up_text(g, "제주도 안내", "제주도 여행에서 흑돼지와 감귤을 꼭 먹어보세요. 렌터카는 미리 예약하세요.")
    assert c.post(f"/chats/{g}/generate", headers=H, json={"kind": "nope"}).status_code == 400
    o = c.post(f"/chats/{g}/generate", headers=H, json={"kind": "summary"}).json()
    assert o["kind"] == "summary" and o["title"].startswith("요약 · ") and o["group_id"] == g
    ans = c.post(f"/chats/{g}/save-answer", headers=H, json={"title": "먹거리 답변", "text": "렌터카 예약 팁 정리"}).json()
    assert c.post(f"/chats/{g}/save-answer", headers=H, json={"title": "x", "text": "  "}).status_code == 400
    lst = c.get("/outputs", headers=H).json()
    assert {x["id"] for x in lst} == {o["id"], ans["id"]} and all(x["group_title"] == "제주 자료방" for x in lst)
    one = c.get(f"/outputs/{o['id']}", headers=H).json()
    assert "흑돼지와 감귤 정리" in one["text"] and one["kind"] == "summary"
    assert c.patch(f"/outputs/{o['id']}", headers=H, json={"title": "제주 요약본"}).json()["title"] == "제주 요약본"
    assert c.patch(f"/outputs/{o['id']}", headers=H, json={"title": " "}).status_code == 400
    # 산출물 검색 (본문·제목), 자료/대화 검색과는 따로
    s = c.get("/search", headers=H, params={"q": "감귤 정리", "scope": "outputs"}).json()["outputs"]
    assert [x["id"] for x in s] == [o["id"]] and "감귤" in s[0]["snippet"]
    assert [x["id"] for x in c.get("/search", headers=H, params={"q": "먹거리", "scope": "outputs"}).json()["outputs"]] == [ans["id"]]
    assert c.get("/search", headers=H, params={"q": "zzzz없는말", "scope": "outputs"}).json() == {"outputs": []}
    # ✍ 이어서 쓰기: 기존 글 뒤에 붙음
    cont = c.post(f"/outputs/{o['id']}/continue", headers=H, json={"instruction": "예시를 더"}).json()
    assert "흑돼지와 감귤 정리" in cont["text"] and cont["text"].endswith("새로 이어진 내용입니다") and "✍ 이어쓰기" in cont["text"]
    assert c.get(f"/outputs/{o['id']}", headers=H).json()["text"] == cont["text"]  # 저장됨
    assert c.post(f"/outputs/{o['id']}/continue", headers=H, json={}).status_code == 200  # 지시 없이도 가능
    # 💬 대화로 이어가기: 글을 자료로 넣은 새 대화방 → 다시 눌러도 같은 방
    r1 = c.post(f"/outputs/{o['id']}/open-chat", headers=H).json()
    r2 = c.post(f"/outputs/{o['id']}/open-chat", headers=H).json()
    assert r1["created"] is True and r2 == {"group_id": r1["group_id"], "created": False}
    cg = r1["group_id"]
    assert [d["title"] for d in c.get(f"/chats/{cg}/docs", headers=H).json()] == ["📦 제주 요약본"]
    assert c.get(f"/chats/{cg}", headers=H).json()["title"] == "이어가기 · 제주 요약본"
    assert c.post(f"/chats/{cg}/ask", headers=H, json={"question": "요약 더 알려줘"}).json()["sources"][0]["title"] == "📦 제주 요약본"
    assert c.get("/chats", headers=H).json()[0]["title"] == "이어가기 · 제주 요약본"  # 직접 지은 제목은 자동으로 안 바뀜
    # 홈 검색: 대화 제목·채팅·자료
    h = c.get("/search", headers=H, params={"q": "렌터카"}).json()
    assert any(d["group_id"] == g for d in h["docs"]) and h["groups"] == []
    assert any(x["id"] == cg for x in c.get("/search", headers=H, params={"q": "이어가기"}).json()["groups"])
    assert any(m["group_id"] == cg for m in c.get("/search", headers=H, params={"q": "요약 더"}).json()["chats"])
    assert c.get("/search", headers=H, params={"q": " "}).json() == {"groups": [], "chats": [], "docs": []}
    # 대화를 지워도 산출물은 남고 '삭제된 대화'로 표시
    c.delete(f"/chats/{g}", headers=H)
    left = {x["id"]: x for x in c.get("/outputs", headers=H).json()}
    assert left[o["id"]]["group_title"] == "(삭제된 대화)" and left[o["id"]]["group_alive"] is False
    assert c.delete(f"/outputs/{ans['id']}", headers=H).json() == {"ok": True}
    assert c.delete(f"/outputs/{ans['id']}", headers=H).status_code == 404
    assert c.get("/outputs/onotexist", headers=H).status_code == 404
    c.delete(f"/chats/{cg}", headers=H); c.delete(f"/outputs/{o['id']}", headers=H)
    assert c.get("/outputs", headers=H).json() == []


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
    # 실제 후보: 안정판 2.5 flash가 있으면 맨 앞(새 3.x 모델이 자주 막혀서)
    saved = (llm.list_models, settings.gemini_chat_model)
    llm.list_models = lambda: models; settings.gemini_chat_model = ""
    try:
        assert llm.chat_model() == ["gemini-2.5-flash", "gemini-3.5-flash", "gemini-3.1-flash-lite"]
    finally:
        llm.list_models, settings.gemini_chat_model = saved


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


def test_gemini_timeout_is_friendly_not_500():
    """Gemini가 응답하지 않아도 500이 아니라 안내 메시지(504)가 나오고, 멈춘 모델 다음 후보로 넘어간다."""
    import httpx
    from fastapi import HTTPException

    saved = (llm._send, llm.chat_model, llm._headers)
    llm._headers = lambda: {"x-goog-api-key": "k", "Content-Type": "application/json"}
    real_request = httpx.request

    def timeout_req(*a, **kw):
        raise httpx.ReadTimeout("read timed out")

    httpx.request = timeout_req
    try:
        llm._send("POST", "models/x:generateContent", {})
        raise AssertionError("시간 초과인데 통과함")
    except HTTPException as e:
        assert e.status_code == 504 and "시간 안에" in e.detail
    finally:
        httpx.request = real_request

    llm.chat_model = lambda: ["slow", "fast"]
    calls = []

    def fake_send(method, path, json=None, timeout=40):
        calls.append(path)
        if "slow" in path:
            raise HTTPException(504, "시간 초과")
        return {"candidates": [{"content": {"parts": [{"text": "OK"}]}}]}

    llm._send = fake_send
    try:
        assert REAL_COMPLETE("s", "u") == "OK" and len(calls) == 2
        llm._send = lambda *a, **k: (_ for _ in ()).throw(HTTPException(504, "시간 초과"))
        try:
            REAL_COMPLETE("s", "u"); raise AssertionError("모두 실패인데 통과함")
        except HTTPException as e:
            assert e.status_code == 504
    finally:
        llm._send, llm.chat_model, llm._headers = saved


def test_overloaded_model_is_skipped_next_time():
    """503으로 막힌 최신 모델은 다음 질문에서 건너뛰고, 성공한 모델을 먼저 쓴다."""
    from fastapi import HTTPException

    saved = (llm._send, llm.chat_model)
    llm.chat_model = lambda: ["new-3.8", "mid-3.7", "ok-3.6"]
    llm._bad_until.clear(); llm._last_ok.clear()
    calls = []

    def fake_send(method, path, json=None, timeout=40):
        calls.append(path.split("/")[1].split(":")[0])
        if "ok-3.6" not in path:
            raise HTTPException(502, "Gemini 오류 503")
        return {"candidates": [{"content": {"parts": [{"text": "OK"}]}}]}

    llm._send = fake_send
    try:
        assert REAL_COMPLETE("s", "u") == "OK" and calls == ["new-3.8", "mid-3.7", "ok-3.6"]
        calls.clear()
        assert REAL_COMPLETE("s", "u") == "OK" and calls == ["ok-3.6"]  # 막힌 모델은 건너뜀
    finally:
        llm._send, llm.chat_model = saved
        llm._bad_until.clear(); llm._last_ok.clear()


def test_thinking_is_kept_short():
    """Gemini 3 계열은 생각을 low로 줄여 빠르게 답하고, 설정을 거부하면 설정 없이 다시 시도한다."""
    from fastapi import HTTPException

    assert llm._gen_config("gemini-3.8-flash") == {"thinkingConfig": {"thinkingLevel": "low"}}
    assert llm._gen_config("gemini-2.5-flash") == {"thinkingConfig": {"thinkingBudget": 0}}
    assert "thinkingConfig" not in llm._gen_config("gemini-1.5-flash")
    saved = (llm._send, llm.chat_model)
    llm.chat_model = lambda: ["gemini-3.8-flash"]
    llm._bad_until.clear(); llm._last_ok.clear()
    seen = []

    def fake_send(method, path, json=None, timeout=40):
        seen.append(json["generationConfig"])
        if json["generationConfig"]:
            raise HTTPException(502, "Gemini 오류 400: thinking level not supported")
        return {"candidates": [{"content": {"parts": [{"text": "OK"}]}}]}

    llm._send = fake_send
    try:
        assert REAL_COMPLETE("s", "u") == "OK"
        assert seen == [{"thinkingConfig": {"thinkingLevel": "low"}}, {}]
    finally:
        llm._send, llm.chat_model = saved
        llm._bad_until.clear(); llm._last_ok.clear()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("통과:", name)
