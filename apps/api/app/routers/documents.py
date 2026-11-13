from fastapi import APIRouter, File, HTTPException, UploadFile

from app import store
from app.config import settings
from app.services import llm
from app.services.chunking import split_text
from app.services.extract import extract_text

router = APIRouter(prefix="/documents", tags=["documents"])


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
    chunks = split_text(text, settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        raise HTTPException(400, "글을 찾지 못했습니다 (스캔 이미지 PDF는 아직 지원하지 않아요)")
    embeddings = llm.embed(chunks)
    doc_id = store.add_document(file.filename or "untitled", chunks, embeddings)
    return {"id": doc_id, "title": file.filename, "chunk_count": len(chunks)}


@router.get("")
def list_all():
    return store.list_documents()


@router.delete("/{doc_id}")
def delete(doc_id: int):
    if not store.delete_document(doc_id):
        raise HTTPException(404, "자료를 찾을 수 없습니다")
    return {"ok": True}
