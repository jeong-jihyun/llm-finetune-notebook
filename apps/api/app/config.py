import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=(".env", "../../.env"), extra="ignore")

    gemini_api_key: str = ""
    gemini_chat_model: str = ""   # 비우면 사용 가능한 무료용 flash 모델을 자동 선택
    gemini_embed_model: str = ""  # 비우면 사용 가능한 embedding 모델을 자동 선택
    app_password: str = ""        # 이 앱에 들어오는 비밀번호 (무료 사용량 보호)
    blob_read_write_token: str = ""  # Vercel Blob 연결 시 자동 주입됨
    data_dir: str = "data"        # Blob 토큰이 없을 때(내 PC 시험용) 저장 폴더
    embed_dim: int = 768
    chunk_size: int = 800
    chunk_overlap: int = 100
    top_k: int = 5
    max_upload_mb: int = 4        # Vercel 요청 크기 한도(4.5MB) 때문
    max_chunks: int = 400         # 자료 1개당 조각 수 한도 (무료 사용량 보호)


settings = Settings()

# Vercel Blob SDK는 환경변수에서 토큰을 읽으므로, .env에 있던 값도 환경변수로 맞춰 준다.
if settings.blob_read_write_token:
    os.environ.setdefault("BLOB_READ_WRITE_TOKEN", settings.blob_read_write_token)
