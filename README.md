# llm-finetune-notebook

자료를 올리면 바로 질문할 수 있고(노트북LM 방식), 버튼 한 번으로 그 자료를 **내 모델로 학습(파인튜닝)** 시키는 웹/모바일 앱.

## 한눈에 보기

| 단계 | 하는 일 | 걸리는 시간 |
|---|---|---|
| 1단계 즉시 사용 | 업로드 → 글 추출 → 조각내기 → 벡터 저장 → 출처 포함 답변 (RAG) | 수 초~수 분 |
| 2단계 깊게 학습 | 자료로 Q&A 학습데이터 자동 생성 → 파인튜닝 → 내 모델로 교체 (무료 방법 조사 중) | 수 분~수 시간 |

> 노트북LM은 "학습"이 아니라 "검색해서 답하기"입니다. 사실 질문은 RAG, 말투·전문용어는 파인튜닝이 맡도록 역할을 나눕니다.

## 구조 (전부 Vercel 안에서, 무료)

```
apps/api/public/index.html  반응형 화면 (자료함 · 채팅 · 점검, 폰/PC 같은 화면)
apps/api/app                FastAPI: 업로드·검색·답변
저장소                      Vercel Blob (비공개)  ← Supabase 안 씀
AI                          Google Gemini API 무료 등급 (임베딩 + 답변)
```

- 이 주소를 아는 누구나 내 무료 사용량을 쓰지 못하게 **비밀번호(APP_PASSWORD)** 로 막습니다.
- Gemini 무료 등급은 사용 횟수 제한이 있고, 입력 내용이 Google 서비스 개선에 쓰일 수 있으니 민감한 자료는 올리지 마세요.
- 자료 1개당 글 약 32만 자(조각 400개), 파일 4MB까지 (Vercel 요청 크기 한도).

## 내 PC에서 시험 (선택, Windows PowerShell)

```powershell
cd apps\api
python -m venv .venv
.\.venv\Scripts\Activate.ps1      # 오류 나면: Set-ExecutionPolicy -Scope Process Bypass
pip install -r requirements.txt
Copy-Item ..\..\.env.example .env   # .env 에 GEMINI_API_KEY, APP_PASSWORD 입력
uvicorn app.main:app --reload --port 8000
```
http://localhost:8000 (Blob 토큰이 없으면 `apps/api/data`에 저장) · 테스트: `python tests/test_flow.py`

## Vercel 배포
Vercel 프로젝트의 Root Directory를 `apps/api`로 두고, 환경변수 `GEMINI_API_KEY`·`APP_PASSWORD`를 넣고, Blob 저장소를 프로젝트에 연결합니다. 화면의 **점검** 탭에서 저장소·Gemini 연결을 확인할 수 있습니다.

> 2단계(학습 데이터 생성·파인튜닝)는 무료로 가능한 방법을 조사한 뒤 진행합니다.

## 개발 순서

1. ✅ RAG 채팅 MVP (업로드 → 질문 → 출처 표시) — 로컬 저장 버전
2. 학습 데이터 생성 + 파인튜닝 작업 관리
3. 학습 모델 연결 + 기본/학습 모델 답변 비교 화면

자세한 내용은 [docs/design.md](docs/design.md).
