import io

from openai import OpenAI

from app.finetune.base import FineTuneProvider, JobStatus

_STATUS_MAP = {
    "validating_files": "queued",
    "queued": "queued",
    "running": "running",
    "succeeded": "succeeded",
    "failed": "failed",
    "cancelled": "cancelled",
}


class OpenAIProvider(FineTuneProvider):
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    def upload_dataset(self, jsonl: str) -> str:
        f = self.client.files.create(
            file=("train.jsonl", io.BytesIO(jsonl.encode("utf-8"))),
            purpose="fine-tune",
        )
        return f.id

    def start_job(self, file_id: str, base_model: str) -> str:
        job = self.client.fine_tuning.jobs.create(training_file=file_id, model=base_model)
        return job.id

    def get_job(self, provider_job_id: str) -> JobStatus:
        job = self.client.fine_tuning.jobs.retrieve(provider_job_id)
        err = job.error.message if getattr(job, "error", None) else None
        return JobStatus(
            provider_job_id=job.id,
            status=_STATUS_MAP.get(job.status, "running"),
            model_id=job.fine_tuned_model,
            error=err,
        )
