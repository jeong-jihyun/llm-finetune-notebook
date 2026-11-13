import hmac

from fastapi import Header, HTTPException

from app.config import settings


def require_password(x_app_password: str = Header(default="")) -> None:
    """공개 주소에서 남이 내 무료 사용량·자료를 쓰지 못하게 막는 비밀번호 확인."""
    if not settings.app_password:
        raise HTTPException(500, "서버에 APP_PASSWORD가 설정되지 않았습니다")
    if not hmac.compare_digest(x_app_password.encode(), settings.app_password.encode()):
        raise HTTPException(401, "비밀번호가 맞지 않습니다")
