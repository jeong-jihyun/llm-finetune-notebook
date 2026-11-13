from pathlib import Path

from fastapi import Depends, FastAPI
from fastapi.responses import FileResponse

from app import storage
from app.auth import require_password
from app.routers import chat, documents
from app.services import llm

app = FastAPI(title="llm-finetune-notebook API")

app.include_router(documents.router)
app.include_router(chat.router)

PUBLIC = Path(__file__).resolve().parent.parent / "public"


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/diag", dependencies=[Depends(require_password)])
def diag():
    """시스템 점검: 저장소와 Gemini 연결 상태 (화면의 '시스템 점검' 버튼)."""
    return {"storage": storage.check(), "gemini": llm.check()}


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(PUBLIC / "index.html")
