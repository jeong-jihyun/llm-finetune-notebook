from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.config import settings
from app.finetune import get_provider
from app.services.dataset import to_chat_jsonl

router = APIRouter(prefix="/finetune", tags=["finetune"])


class ExampleIn(BaseModel):
    question: str
    answer: str


class StartJobIn(BaseModel):
    examples: list[ExampleIn]


@router.post("/jobs")
def start_job(body: StartJobIn):
    """Q&A 목록을 JSONL로 만들어 외부 API에 올리고 학습을 시작한다. (DB 저장·인증은 다음 단계)"""
    jsonl = to_chat_jsonl([e.model_dump() for e in body.examples])
    if not jsonl:
        raise HTTPException(400, "학습에 쓸 수 있는 예시가 없습니다")
    provider = get_provider()
    file_id = provider.upload_dataset(jsonl)
    job_id = provider.start_job(file_id, settings.finetune_base_model)
    return {"provider_job_id": job_id, "example_count": jsonl.count("\n") + 1}


@router.get("/jobs/{provider_job_id}")
def get_job(provider_job_id: str):
    s = get_provider().get_job(provider_job_id)
    return {"status": s.status, "model_id": s.model_id, "error": s.error}
