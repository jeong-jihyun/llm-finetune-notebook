from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import finetune

app = FastAPI(title="llm-finetune-notebook API")

# 개발 중에는 로컬 웹만 허용. 배포 시 Vercel 주소를 추가한다.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(finetune.router)


@app.get("/health")
def health():
    return {"ok": True}
