from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel

from app import storage
from app.auth import require_password
from app.config import settings
from app.services import llm
from app.services.chunking import split_text
from app.services.extract import extract_text

router = APIRouter(prefix="/documents", tags=["documents"], dependencies=[Depends(require_password)])


def _ingest(title: str, text: str) -> dict:
    chunks = split_text(text, settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        raise HTTPException(400, "글을 찾지 못했습니다 (스캔 이미지 PDF는 아직 지원하지 않아요)")
    if len(chunks) > settings.max_chunks:
        raise HTTPException(
            413,
            f"자료가 너무 깁니다 (조각 {len(chunks)}개). 무료 사용량 보호를 위해 약 "
            f"{settings.max_chunks * settings.chunk_size:,}자(조각 {settings.max_chunks}개)까지만 올릴 수 있어요. 나눠서 올려 주세요.",
        )
    doc_id = storage.add_document(title, chunks, llm.embed(chunks))
    return {"id": doc_id, "title": title, "chunk_count": len(chunks)}


@router.post("")
async def upload(file: UploadFile = File(...)):
    data = await file.read()
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"파일이 너무 큽니다 (최대 {settings.max_upload_mb}MB)")
    try:
        text = extract_text(file.filename or "", data)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception:
        raise HTTPException(400, "파일을 읽을 수 없습니다 (손상되었거나 암호가 걸린 파일일 수 있어요)")
    return _ingest(file.filename or "untitled", text)


class TextIn(BaseModel):
    title: str
    text: str


@router.post("/text")
def upload_text(body: TextIn):
    if len(body.text.encode("utf-8")) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"글이 너무 깁니다 (최대 {settings.max_upload_mb}MB)")
    return _ingest(body.title.strip() or "붙여넣은 글", body.text)


@router.get("")
def list_all():
    return storage.list_documents()


@router.delete("/{doc_id}")
def delete(doc_id: str):
    if not storage.delete_document(doc_id):
        raise HTTPException(404, "자료를 찾을 수 없습니다")
    return {"ok": True}
