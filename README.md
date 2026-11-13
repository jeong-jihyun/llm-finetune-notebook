# llm-finetune-notebook

자료를 올리면 바로 질문할 수 있고(노트북LM 방식), 버튼 한 번으로 그 자료를 **내 모델로 학습(파인튜닝)** 시키는 웹/모바일 앱.

## 한눈에 보기

| 단계 | 하는 일 | 걸리는 시간 |
|---|---|---|
| 1단계 즉시 사용 | 업로드 → 글 추출 → 조각내기 → 벡터 저장 → 출처 포함 답변 (RAG) | 수 초~수 분 |
| 2단계 깊게 학습 | 자료로 Q&A 학습데이터 자동 생성 → 외부 파인튜닝 API로 학습 → 내 모델로 교체 | 수 분~수 시간 |

> 노트북LM은 "학습"이 아니라 "검색해서 답하기"입니다. 사실 질문은 RAG, 말투·전문용어는 파인튜닝이 맡도록 역할을 나눕니다.

## 구조

```
apps/web    Next.js(반응형 + PWA)  → Vercel 배포
apps/api    FastAPI                → 업로드·RAG·학습 작업 관리
supabase/   DB 스키마(pgvector)·로그인·파일 저장
docs/       설계 문서
```

## 시작하기 (Windows PowerShell)

```powershell
cd apps\api
python -m venv .venv
.\.venv\Scripts\Activate.ps1      # 오류 나면: Set-ExecutionPolicy -Scope Process Bypass
pip install -r requirements.txt
Copy-Item ..\..\.env.example .env   # apps\api\.env 에 OPENAI_API_KEY 입력
uvicorn app.main:app --reload --port 8000
```

브라우저에서 http://localhost:8000 → **자료함**에서 txt/md/pdf 올리기 → **채팅**에서 질문.
(같은 와이파이의 폰에서는 `uvicorn ... --host 0.0.0.0` 로 켜고 `http://PC주소:8000` 접속)

> 1단계는 시험하기 쉽게 로컬 SQLite 저장 + 서버가 직접 보여주는 반응형 화면입니다.
> Supabase(pgvector)·Next.js로 바꾸는 것은 이후 단계입니다. 자료는 `apps/api/data/`에 저장됩니다(git 제외).

## 개발 순서

1. ✅ RAG 채팅 MVP (업로드 → 질문 → 출처 표시) — 로컬 저장 버전
2. 학습 데이터 생성 + 파인튜닝 작업 관리
3. 학습 모델 연결 + 기본/학습 모델 답변 비교 화면

자세한 내용은 [docs/design.md](docs/design.md).
