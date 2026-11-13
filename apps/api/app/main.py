from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app import store
from app.routers import chat, documents, finetune

app = FastAPI(title="llm-finetune-notebook API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router)
app.include_router(chat.router)
app.include_router(finetune.router)

STATIC = Path(__file__).parent / "static"


@app.on_event("startup")
def _startup():
    store.init()


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")
