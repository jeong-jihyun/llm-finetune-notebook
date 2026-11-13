"""글을 조각내고 임베딩해서 그룹 자료로 저장한다 (파일 업로드·붙여넣기·'대화로 이어가기'가 같이 씀)."""
from fastapi import HTTPException

from app import storage
from app.config import settings
from app.services import llm
from app.services.chunking import split_text


def ingest(gid: str, title: str, text: str) -> dict:
    chunks = split_text(text, settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        raise HTTPException(400, "글을 찾지 못했습니다 (스캔 이미지 PDF는 아직 지원하지 않아요)")
    if len(chunks) > settings.max_chunks:
        raise HTTPException(
            413,
            f"자료가 너무 깁니다 (조각 {len(chunks)}개). 무료 사용량 보호를 위해 약 "
            f"{settings.max_chunks * settings.chunk_size:,}자(조각 {settings.max_chunks}개)까지만 올릴 수 있어요. 나눠서 올려 주세요.",
        )
    return storage.add_document(gid, title, chunks, llm.embed(chunks))
