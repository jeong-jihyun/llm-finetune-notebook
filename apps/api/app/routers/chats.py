from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel

from app import storage
from app.auth import require_password
from app.config import settings
from app.services import chats, generate, ingest, rag, search as search_svc
from app.services.extract import extract_text

router = APIRouter(prefix="/chats", tags=["chats"], dependencies=[Depends(require_password)])
search_router = APIRouter(prefix="/search", tags=["search"], dependencies=[Depends(require_password)])


class TitleIn(BaseModel):
    title: str = ""


MAX_IMAGES = 4
IMG_MIMES = {"image/jpeg", "image/png", "image/webp", "image/gif"}
MAX_IMG_B64 = 1_400_000      # 이미지 1장 (base64 글자 수)
MAX_TOTAL_B64 = 3_600_000    # 이미지 전체 합 — Vercel 요청 한도(4.5MB) 안에 들어가게
MAX_THUMB = 60_000


class ImageIn(BaseModel):
    mime: str
    data: str


class AskIn(BaseModel):
    question: str = ""
    images: list[ImageIn] = []
    thumbs: list[str] = []


class TextIn(BaseModel):
    title: str
    text: str


class KindIn(BaseModel):
    kind: str


class SaveAnswerIn(BaseModel):
    title: str = ""
    text: str


# ── 대화 그룹 ────────────────────────────────────────────────────────────
@router.get("")
def list_groups():
    return chats.list_groups()


@router.post("")
def create(body: TitleIn):
    return chats.create_group(body.title)


@router.patch("/{gid}")
def rename(gid: str, body: TitleIn):
    return chats.rename_group(gid, body.title)


@router.delete("/{gid}")
def delete(gid: str):
    chats.delete_group(gid)  # 대화 기록과 그 대화의 자료를 지움 (산출물은 남음)
    return {"ok": True}


@router.get("/{gid}")
def messages(gid: str):
    return {"id": gid, "title": chats.get_group(gid)["title"], "messages": chats.get_messages(gid)}


@router.post("/{gid}/ask")
def ask(gid: str, body: AskIn):
    q = body.question.strip()
    if not q and not body.images:
        raise HTTPException(400, "질문을 입력해 주세요")
    if len(body.images) > MAX_IMAGES:
        raise HTTPException(400, f"이미지는 한 번에 {MAX_IMAGES}장까지 보낼 수 있어요")
    images = []
    for im in body.images:
        if im.mime not in IMG_MIMES:
            raise HTTPException(400, "이미지는 jpg, png, webp, gif만 가능해요")
        if len(im.data) > MAX_IMG_B64:
            raise HTTPException(413, "이미지가 너무 커요. 더 작은 이미지로 올려 주세요")
        images.append({"mime": im.mime, "data": im.data})
    if sum(len(i["data"]) for i in images) > MAX_TOTAL_B64:
        raise HTTPException(413, "이미지 용량 합이 너무 커요. 장수를 줄여 주세요")
    thumbs = [t for t in body.thumbs[:MAX_IMAGES] if t.startswith("data:image/") and len(t) <= MAX_THUMB]
    chats.get_group(gid)  # 그룹이 있는지 먼저 확인
    q = q or "이 이미지를 설명해 주세요."
    result = rag.answer(gid, q, images or None)
    chats.add_exchange(gid, q, result, thumbs or None)  # 답변이 성공했을 때만 기록 (+ 첫 질문이면 제목 자동 생성)
    return result


# ── 이 대화의 자료 ───────────────────────────────────────────────────────
@router.get("/{gid}/docs")
def list_docs(gid: str):
    chats.get_group(gid)
    return storage.list_documents(gid)


@router.post("/{gid}/docs")
async def upload(gid: str, file: UploadFile = File(...)):
    chats.get_group(gid)
    data = await file.read()
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"파일이 너무 큽니다 (최대 {settings.max_upload_mb}MB)")
    try:
        text = extract_text(file.filename or "", data)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception:
        raise HTTPException(400, "파일을 읽을 수 없습니다 (손상되었거나 암호가 걸린 파일일 수 있어요)")
    return ingest.ingest(gid, file.filename or "untitled", text)


@router.post("/{gid}/docs/text")
def upload_text(gid: str, body: TextIn):
    chats.get_group(gid)
    if len(body.text.encode("utf-8")) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"글이 너무 깁니다 (최대 {settings.max_upload_mb}MB)")
    return ingest.ingest(gid, body.title.strip() or "붙여넣은 글", body.text)


@router.delete("/{gid}/docs/{doc_id}")
def delete_doc(gid: str, doc_id: str):
    chats.get_group(gid)
    if not storage.delete_document(gid, doc_id):
        raise HTTPException(404, "자료를 찾을 수 없습니다")
    return {"ok": True}


# ── 산출물 만들기 (결과는 📦 산출물 탭으로) ──────────────────────────────
@router.post("/{gid}/generate")
def generate_output(gid: str, body: KindIn):
    return generate.make_output(gid, body.kind)


@router.post("/{gid}/save-answer")
def save_answer(gid: str, body: SaveAnswerIn):
    return generate.save_answer(gid, body.title, body.text)


@search_router.get("")
def search(q: str = "", scope: str = "home"):
    return search_svc.search(q.strip(), scope)
