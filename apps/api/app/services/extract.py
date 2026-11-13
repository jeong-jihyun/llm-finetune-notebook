import io

from pypdf import PdfReader


def extract_text(filename: str, data: bytes) -> str:
    """파일에서 글만 뽑는다. 지원: txt, md, pdf"""
    name = filename.lower()
    if name.endswith((".txt", ".md")):
        for enc in ("utf-8", "cp949"):
            try:
                return data.decode(enc)
            except UnicodeDecodeError:
                continue
        return data.decode("utf-8", errors="ignore")
    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    raise ValueError("지원하지 않는 파일 형식입니다 (txt, md, pdf만 가능)")
