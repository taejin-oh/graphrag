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

---

## 6) QFS batch index 재시작(resume) 사용법

`ltm_qfs` 브랜치의 `scripts/run_qfs_index.py`는 중간 실패 후 이어서 실행할 수 있도록 아래 옵션을 지원합니다.

- `--resume`: `logs/index_status.json`이 `success`인 test_id는 건너뜀
- `--continue-on-error`: 실패해도 다음 test_id 계속 수행
- `--force-clean`: 실행 전 `logs/`, `output/` 삭제 후 재생성

### 기본 JSON 입력(기본값)

- `--input-type` 기본값: `json`
- `--input-file-pattern` 기본값: `.*\.json$`

### 예시

```bash
# 원하는 test_case를 순서대로 실행
python scripts/run_qfs_index.py --test-cases 100K 500K 1M

# 실패 지점부터 이어서, 실패해도 계속 진행
python scripts/run_qfs_index.py --test-case 100K --resume --continue-on-error

# 강제 재실행(로그/출력 초기화)
python scripts/run_qfs_index.py --test-case 100K --force-clean
```

---

## 7) QFS query/aggregate에서 test_id 지정 실행

`scripts/run_qfs_query_and_aggregate.py`는 `test_case` 고정 외에 `test_id`도 임의 지정할 수 있습니다.

- `--test-case`: 특정 케이스만 선택
- `--test-ids`: 해당 케이스 내에서 실행할 test_id 목록 선택
- `--debug`: 질문 단위 진행 로그 출력
- `--show-assembled-context`: `--debug`와 함께 질문별 assembled_context 본문 출력
- 기본값으로 assembled_context는 summary 중심 slim 포맷으로 정리됨
  - `id`, `title`, `nid`, `[Data: ...]` 제거
  - `timeline_events`, `superseded_facts`는 summary 위주로 축약
- `--raw-assembled-context`: 원문 assembled_context를 그대로 저장/출력
- `--max-tokens`: `experimental_context_max_tokens`를 CLI에서 직접 지정

출력 파일:
- `--max-tokens` 미지정 시
  - `results.csv`
  - `results.jsonl` (라인 단위 처리용)
  - `results.json` (들여쓰기 적용, 사람이 읽기 쉬운 포맷)
- `--max-tokens N` 지정 시
  - `results_maxN.csv`
  - `results_maxN.jsonl`
  - `results_maxN.json`

예시:

```bash
# case_a 안에서 001, 003 test_id만 실행
python scripts/run_qfs_query_and_aggregate.py \
  --test-case case_a \
  --test-ids 001 003 \
  --run-id qfs_case_a_sel

# 디버그 로그 + assembled_context 본문 출력
python scripts/run_qfs_query_and_aggregate.py \
  --test-case case_a \
  --test-ids 001 \
  --debug \
  --show-assembled-context

# 토큰 상한을 걸고 실행(파일명에 _max1200 suffix 반영)
python scripts/run_qfs_query_and_aggregate.py \
  --test-case case_a \
  --test-ids 001 \
  --max-tokens 1200 \
  --run-id qfs_case_a_t1200
```

---

## 8) Index + Query(total) 연속 실행 스크립트

`scripts/run_qfs_total_pipeline.py`는 `test_case/test_id` 단위로 아래 순서를 자동 수행합니다.

1. `run_qfs_index.py`
2. `run_qfs_query_and_aggregate.py` (policy 순서대로 반복)

특징:
- 기본 policy 순서: `pyramid -> flat_ranked -> leaf_only -> leaf_then_parent_mix`
- 각 index/query 단위 완료 후 10분(`600초`) 대기 (기본값, 마지막 완료 직후는 대기 없음)
- 실패 대상은 라운드 종료 후 재시도하여 모두 성공할 때까지 반복
- `--max-rounds`로 재시도 상한 설정 가능 (`0`이면 무제한)
- `--max-tokens` 지정 시 하위 `run_qfs_query_and_aggregate.py` 호출로 그대로 전달
- 진행 로그는 `qfs_log/<run_id>/progress_log.txt`에 기록되고 콘솔에는 노란색으로 출력

예시:

```bash
python scripts/run_qfs_total_pipeline.py \
  --test-case case_a \
  --test-ids 001 003 \
  --max-tokens 1200 \
  --run-id qfs_total_case_a_t1200
```

---

## 9) query/aggregate 멈춤 지점 디버그

`scripts/debug_qfs_query_runner.py`를 사용하면 질문 단위 timeout과 단계별 로그로 어디서 멈추는지 확인할 수 있습니다.

```bash
python scripts/debug_qfs_query_runner.py \
  --test-case 100K \
  --test-ids 7 \
  --max-tokens 1000 \
  --per-query-timeout 120
```

timeout이 발생해도 계속 진행하려면:

```bash
python scripts/debug_qfs_query_runner.py \
  --test-case 100K \
  --test-ids 7 \
  --max-tokens 1000 \
  --per-query-timeout 120 \
  --continue-on-timeout
```

질문별 `assembled_context` 본문까지 출력하려면:

```bash
python scripts/debug_qfs_query_runner.py \
  --test-case 100K \
  --test-ids 7 \
  --max-tokens 1000 \
  --show-assembled-context
```
