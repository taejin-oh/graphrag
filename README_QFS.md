# QFS 최소 Assembled Context 출력 가이드

## 목적
- Query 단계에서 아래 3개 필수 결과를 안정적으로 출력합니다.
  - `assembled_context`
  - `assembled_context_tokens`
  - `selected_community_ids`
- 옵션으로 최종 LLM 응답 생성을 건너뛰고 context만 확인할 수 있습니다.

## 변경 이유
- 실험/검증 시 최종 답변보다 “LLM에 들어간 컨텍스트” 자체를 빠르게 확인할 필요가 있습니다.
- 불필요한 LLM 호출을 줄여 디버깅 반복 속도를 높일 수 있습니다.

## 변경 사항 요약
- Local query CLI 옵션 추가
  - `--show-assembled-context`
  - `--context-only`
- LocalSearch에 `context_only` 분기 추가
  - context 구성 후 LLM completion 호출을 건너뜁니다.
- local search 결과에서 `context_chunks`를 함께 전달
- CLI에서 최소 payload(`minimal_assembled_context`)를 구성/출력

## 사용 방법

### 1) 기존처럼 질의 + 필수 context 출력
```bash
python -m graphrag query \
  --method local \
  --query "Who is X?" \
  --show-assembled-context
```

### 2) 최종 LLM 호출 없이 context만 확인
```bash
python -m graphrag query \
  --method local \
  --query "Who is X?" \
  --context-only \
  --show-assembled-context
```

## 출력 예시
```text
===== assembled_context =====
selected_community_ids: ['10', '20']
assembled_context_tokens: 321
-----Reports-----
id|title|summary
10|...
20|...
===== /assembled_context =====
```

## 기대 효과
- Query 디버깅 시 컨텍스트 검증 시간을 단축
- LLM 비용/지연 최소화(`--context-only`)
- 필수 지표(본문/토큰/선택 커뮤니티 ID) 표준화

## QFS 스크립트 실행 가이드

### 1) 인덱싱
```bash
python scripts/run_qfs_index.py \
  --input-chat-root input_chat \
  --retry-count 3 \
  --retry-sleep-seconds 60
```

- 인덱싱 시 vector store(`lancedb`)는 각 `test_id/output/lancedb` 경로에 저장됩니다.
- 따라서 test_id 간 vector index가 서로 덮어쓰이지 않습니다.

### 2) Query + 집계(JSON만 출력)
```bash
python scripts/run_qfs_query_and_aggregate.py \
  --input-chat-root input_chat \
  --run-id qfs_run_001 \
  --max-tokens 4000 \
  --retry-count 3 \
  --retry-sleep-seconds 60
```

- Query 시에도 동일한 `test_id/output/lancedb`를 사용합니다.

- 결과 파일: `qfs_log/<run-id>/<condition>/results(_maxN).json`
- 필수 필드:
  - `assembled_context`
  - `assembled_context_tokens`
  - `selected_community_ids`
- 추가 필드(가독성/운영용):
  - `condition_id`
  - `community_policy`
  - `selected_community_context`
  - `selected_community_summaries`
  - `selected_community_current_states`
  - `selected_community_timeline_events`
  - `max_context_tokens`
  - `configured_use_community_summary`
  - `detected_community_context_column`
  - `use_community_summary_applied`
  - `error`

### 3) 전체 파이프라인(index -> query)
```bash
python scripts/run_qfs_total_pipeline.py \
  --input-chat-root input_chat \
  --run-id qfs_total_001 \
  --sleep-seconds 60
```

## 지원 정책 안내
- 현재 코드 기준으로 QFS 평가 스크립트에서 지원하는 정책 값은 `default` 하나입니다.
- 기존 `pyramid`, `flat_ranked`, `leaf_only`, `leaf_then_parent_mix`는 현재 브랜치 코드에 해당 구현이 없어 제거했습니다.

## 자주 묻는 질문
- `settings.yml`의 `local_search.use_community_summary: true` 설정으로 local query에서 community summary 우선 사용이 가능합니다.
- QFS 스크립트는 이 설정과 별개로 `selected_community_summaries` 필드에 선택된 커뮤니티 요약을 개별 항목으로 함께 기록합니다.
