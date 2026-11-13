# CLAUDE.md

- 한국어로 답하고, 쉽고 단순하게 설명한다.
- 사용자 환경은 Windows PowerShell 기준.
- 구조: apps/web(Next.js), apps/api(FastAPI), supabase/(스키마), docs/design.md가 기준 설계.
- 파인튜닝은 외부 API 방식. 공급자는 `apps/api/app/finetune/`의 인터페이스로 교체 가능하게 유지.
- 새 테이블은 반드시 user_id + RLS 적용.
- GitHub push 전에는 먼저 사용자에게 보여 주고 확인받는다.
