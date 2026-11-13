# 설계 문서 — llm-finetune-notebook

## 1. 목표
사용자가 자료(PDF·문서·URL)를 넣으면 ① 바로 질문·답변이 되고 ② 원하면 그 자료로 모델을 학습시켜 "내 모델"을 얻는다. 웹과 모바일 모두 같은 화면 코드로 대응한다.

## 2. 생각의 흐름 (왜 이 구조인가)
1. "자료 넣으면 학습"이 가장 원하는 모습이다.
2. 그런데 진짜 학습은 오래 걸리고 비용이 든다 → 올리자마자 쓸 수 없다.
3. 그래서 **먼저 검색 방식(RAG)** 으로 즉시 쓰게 하고, **학습은 뒤에서 돌리다가 끝나면 교체**한다.
4. GPU를 직접 관리하면 복잡하다 → **외부 파인튜닝 API**를 쓴다. 공급자는 바꿀 수 있게 인터페이스로 분리한다.

## 3. 전체 구조

```
[웹/모바일(Next.js PWA)]
        │ HTTPS
        ▼
[FastAPI] ──► Supabase(Auth·Storage·Postgres+pgvector)
   │  ├─ /documents  업로드·처리
   │  ├─ /chat       RAG 질의응답 (학습 모델 있으면 그 모델 사용)
   │  └─ /finetune   학습 데이터 생성·학습 작업 시작·상태 조회
   ▼
[외부 파인튜닝 API]  (기본: OpenAI, 교체 가능)
```

## 4. 데이터 흐름

**업로드(1단계)**
파일 → Storage 저장 → 글 추출 → 조각(청크, 약 800자·겹침 100자) → 임베딩 → `chunks`(pgvector) 저장

**질문**
질문 임베딩 → 비슷한 청크 상위 k개 검색 → 모델에 "이 자료만 근거로 답해" 프롬프트 → 답변 + 출처(문서·청크)

**학습(2단계)**
1. 청크마다 LLM으로 Q&A 생성 (`training_examples`)
2. 중복·짧은 것·근거 없는 것 필터
3. JSONL 변환 → 파인튜닝 API 업로드 → 작업 생성
4. 주기적으로 상태 확인(`queued → running → succeeded/failed`)
5. 성공하면 `models`에 모델 ID 저장 → 채팅에서 "학습 모델" 선택 가능

## 5. 화면 (모바일 우선, 데스크톱은 같은 화면을 넓게)
1. **자료함** — 업로드, 처리 상태(대기·처리중·완료)
2. **채팅** — 답변 아래 출처 칩, 탭하면 원문 조각 표시
3. **학습** — 학습 데이터 미리보기, 예상 비용, 시작 버튼, 진행률
4. **내 모델** — 모델 목록, 같은 질문에 기본/학습 모델 답변 나란히 비교

모바일은 하단 탭 4개, 데스크톱은 좌측 사이드바. PWA로 홈 화면에 설치 가능.

## 6. API (초안)

| 메서드 | 경로 | 설명 |
|---|---|---|
| GET | /health | 상태 확인 |
| POST | /documents | 파일 업로드 → 처리 작업 시작 |
| GET | /documents | 내 자료 목록·상태 |
| POST | /chat | `{question, model_id?}` → 답변+출처 |
| POST | /finetune/datasets | 학습 데이터 생성 시작 |
| POST | /finetune/jobs | 학습 작업 시작 |
| GET | /finetune/jobs/{id} | 상태·진행률 |
| GET | /models | 내 모델 목록 |

## 7. DB
`supabase/migrations/0001_init.sql` 참고. 핵심 테이블: `documents`, `chunks`, `training_examples`, `finetune_jobs`, `models`, `messages`. 모든 테이블에 `user_id`와 RLS(본인 것만 접근) 적용.

## 8. 외부 파인튜닝 API 연결
`apps/api/app/finetune/base.py`의 `FineTuneProvider` 인터페이스(`upload_dataset`, `start_job`, `get_job`)를 구현한 공급자를 환경변수 `FINETUNE_PROVIDER`로 선택한다. 기본 구현은 OpenAI(`openai_provider.py`). 기본 모델 이름은 `FINETUNE_BASE_MODEL`로 지정하며, **파인튜닝이 지원되는 모델인지 공급자 문서에서 확인 후 사용**한다.

## 9. 위험과 대응
- 자료가 적으면 학습해도 사실을 잘 못 외움 → 사실 질문은 RAG, 학습 모델은 말투·용어·형식에 활용
- 비용 폭증 → 학습 시작 전 예상 비용 표시, 사용자당 월 학습 횟수 제한
- 개인정보·저작권 → 업로드 시 동의 문구, 자료·모델 삭제 기능, 사용자별 격리(RLS)
- 학습 실패/지연 → 상태 폴링 + 실패 사유 표시, 재시도 버튼

## 10. 개발 단계
| 단계 | 내용 |
|---|---|
| 1 | 업로드·청크·임베딩·RAG 채팅·출처 표시 |
| 2 | Q&A 생성·JSONL·학습 작업·상태 표시 |
| 3 | 학습 모델 연결·비교 화면·비용 제한 |
