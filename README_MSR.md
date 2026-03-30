# README_MSR

이 문서는 `experimental local context` 관련으로 지금까지 반영된 수정사항을 한국어로 정리한 운영 메모입니다.

## 1) 이번에 반영된 핵심 변경점

### 1-1. CLI에서 assembled_context 즉시 출력 지원

`graphrag query` 명령에 아래 옵션이 추가되었습니다.

- `--show-assembled-context`
- `--hide-assembled-context`

`--method local`로 실행할 때, `--show-assembled-context`를 켜면 실행 결과와 함께
`experimental_context` payload 안의 핵심 값이 터미널에 출력됩니다.

- `condition_id`
- `assembled_context_tokens`
- `warnings`
- `assembled_context` 본문

### 1-2. streaming / non-streaming 모두 출력 지원

local search 경로에서 streaming, non-streaming 모두 동일하게 `show_assembled_context=True`일 때
payload 출력 함수를 타도록 연결했습니다.

### 1-3. 문서 보강

`docs/experimental_local_context_실사용_가이드.md`에 아래 내용을 추가/보강했습니다.

- 바로 복붙 가능한 `query.txt` 예시
- `RUN_ID` 기반 실행 + 추출 워크플로우
- `graphrag query --method local --show-assembled-context` 사용 예시
- 출력이 길 때 `--hide-assembled-context` 안내
- 출력 예시 블록
- 검증 섹션(실행 확인용 커맨드 예시)

### 1-4. 테스트 추가

`tests/unit/cli/test_query_show_assembled_context.py`를 추가했습니다.

- payload 정상 출력 포맷 검증
- payload 누락 시 메시지 검증
- CLI 플래그 전달(`--show-assembled-context`) 검증

---

## 2) 사용법

## 2-1. settings.yaml 확인

아래 값이 `local_search`에 있어야 실험 컨텍스트 payload를 보기 쉽습니다.

```yaml
local_search:
  experimental_context_mode: true
  experimental_log_context_payload: true
  experimental_community_policy: flat_ranked
  experimental_history_enabled: false
  experimental_covariate_enabled: false
  experimental_context_max_tokens: 1800
  experimental_condition_id: "manual_q001_c01"
  experimental_policy_preserve_mode: fallback
```

## 2-2. local query 실행 시 assembled_context 출력

```bash
graphrag query "Who is Scrooge and what are his main relationships?" \
  --method local \
  --root . \
  --data ./output \
  --show-assembled-context
```

출력이 너무 길면:

```bash
graphrag query "Who is Scrooge and what are his main relationships?" \
  --method local \
  --root . \
  --data ./output \
  --hide-assembled-context
```

## 2-3. 로그에서 payload 확인

```bash
rg "\[LOCAL_CONTEXT_PAYLOAD\]" logs/query.log | tail -n 20
```

---

## 3) 동작 방식(요약)

1. `query` CLI에서 `--method local`로 분기됩니다.
2. `--show-assembled-context` 값이 local search 실행 함수로 전달됩니다.
3. local search 완료 후 `context_data["experimental_context"]`를 읽습니다.
4. payload가 있으면 `assembled_context` 관련 정보를 포맷팅해 stdout에 출력합니다.
5. payload가 없거나 비어 있으면 안내 메시지를 출력합니다.

---

## 4) 검증 커맨드

```bash
# 단위 테스트
PYTHONPATH=packages/graphrag pytest -q tests/unit/cli/test_query_show_assembled_context.py

# 통합 테스트
PYTHONPATH=packages/graphrag pytest -q tests/integration/query/test_experimental_local_context_integration.py
```

---

## 5) 운영 팁

- 16조건(정책 4 × history on/off × covariate on/off)을 직접 돌릴 때는
  `experimental_condition_id`를 매 실행마다 고유하게 넣어두면 추적이 쉬워집니다.
- 실행 중 즉시 비교는 `--show-assembled-context`, 사후 비교는 `query.log` 추출 방식이 편합니다.
