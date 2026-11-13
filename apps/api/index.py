"""Vercel 진입점. (내 PC에서는 `uvicorn index:app` 또는 `uvicorn app.main:app`)"""
from app.main import app  # noqa: F401
