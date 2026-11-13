from app.config import settings
from app.finetune.base import FineTuneProvider


def get_provider() -> FineTuneProvider:
    """환경변수 FINETUNE_PROVIDER에 따라 공급자를 고른다."""
    if settings.finetune_provider == "openai":
        from app.finetune.openai_provider import OpenAIProvider

        return OpenAIProvider(settings.openai_api_key)
    raise ValueError(f"지원하지 않는 공급자: {settings.finetune_provider}")
