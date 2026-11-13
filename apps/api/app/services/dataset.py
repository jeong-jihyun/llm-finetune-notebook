import json


def to_chat_jsonl(examples: list[dict], system_prompt: str = "당신은 사용자의 자료를 바탕으로 답하는 도우미입니다.") -> str:
    """질문/답변 목록을 파인튜닝용 JSONL(한 줄에 대화 하나)로 바꾼다."""
    lines = []
    for ex in examples:
        q, a = (ex.get("question") or "").strip(), (ex.get("answer") or "").strip()
        if len(q) < 5 or len(a) < 5:  # 너무 짧은 것은 버림
            continue
        lines.append(
            json.dumps(
                {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": q},
                        {"role": "assistant", "content": a},
                    ]
                },
                ensure_ascii=False,
            )
        )
    return "\n".join(lines)
