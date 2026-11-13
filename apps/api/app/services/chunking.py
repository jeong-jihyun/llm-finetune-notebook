def split_text(text: str, size: int = 800, overlap: int = 100) -> list[str]:
    """글을 일정 길이로 자르되 앞뒤를 조금 겹쳐서 문맥이 끊기지 않게 한다."""
    if size <= overlap:
        raise ValueError("size는 overlap보다 커야 합니다")
    text = " ".join(text.split())
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start : start + size])
        start += size - overlap
    return chunks
