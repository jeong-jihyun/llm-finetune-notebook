from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=(".env", "../../.env"), extra="ignore")

    supabase_url: str = ""
    supabase_service_key: str = ""
    openai_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"  # 1536차원 (DB 스키마와 일치)
    chat_model: str = "gpt-4o-mini"
    finetune_provider: str = "openai"
    finetune_base_model: str = "gpt-4o-mini-2024-07-18"  # 공급자 문서에서 지원 여부 확인 후 변경
    chunk_size: int = 800
    chunk_overlap: int = 100
    top_k: int = 5
    max_upload_mb: int = 10


settings = Settings()
