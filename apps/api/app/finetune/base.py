from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class JobStatus:
    provider_job_id: str
    status: str  # queued | running | succeeded | failed | cancelled
    model_id: str | None = None  # 성공 시 학습된 모델 ID
    error: str | None = None


class FineTuneProvider(ABC):
    """외부 파인튜닝 API를 바꿔 끼울 수 있게 하는 공통 인터페이스."""

    @abstractmethod
    def upload_dataset(self, jsonl: str) -> str:
        """학습 데이터를 올리고 파일 ID를 돌려준다."""

    @abstractmethod
    def start_job(self, file_id: str, base_model: str) -> str:
        """학습 작업을 시작하고 작업 ID를 돌려준다."""

    @abstractmethod
    def get_job(self, provider_job_id: str) -> JobStatus:
        """작업 상태를 조회한다."""
