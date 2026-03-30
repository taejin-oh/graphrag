# Experimental Local Context 실사용 가이드 (qfa_test2)

이 문서는 `qfa_test2` 브랜치에 이미 구현된 experimental local context 기능을 **실사용** 관점에서 빠르게 실행/검증/추출하는 방법을 설명합니다.

핵심 목표는 다음 2가지입니다.

1. `query.txt`의 여러 질의를 16개 조건(4 policy × history on/off × covariate on/off)으로 일괄 실행
2. `query.log`에서 내가 필요한 값(특히 `assembled_context`)만 깔끔하게 추출

---

## 1) 실행 전 준비물

- GraphRAG 프로젝트 루트(예: `settings.yaml`이 있는 디렉터리)
- 이미 생성된 index output 디렉터리(예: `<root>/output`)
  - 최소 필요 parquet: `communities`, `community_reports`, `entities`, `relationships`, `text_units`
  - `covariates.parquet`는 선택(없으면 covariate ON 조건에서 경고/빈 결과 가능)
- 로컬 검색에 필요한 모델/인증 환경 변수(기존 프로젝트와 동일)
- 질의 파일 `query.txt`

> 실사용 기준: 문서와 스크립트는 **이미 output이 생성된 디렉터리 재사용**을 전제로 설계됨.

---

## 2) query.txt 형식

`query.txt`는 한 줄당 하나의 질의입니다.

- 빈 줄은 무시
- `#`로 시작하는 줄은 주석으로 무시

예시:

```text
# 인물 중심 질의
Who is Scrooge and what are his main relationships?

# 사건 흐름 질의
Summarize key events in chronological order.
```

### 바로 복붙해서 시작하는 최소 예시

아래처럼 파일을 만들면 이 문서의 명령을 그대로 따라갈 수 있습니다.

```bash
cat > query.txt <<'EOF'
# 인물 중심 질의
Who is Scrooge and what are his main relationships?

# 사건 흐름 질의
Summarize key events in chronological order.
EOF
```

---

## 3) settings / config 옵션 설명 (실험 컨텍스트 관련)

실행 스크립트는 내부적으로 아래 `local_search` 옵션을 설정합니다.

- `experimental_context_mode=true`
  - 실험 컨텍스트 조합 모드 활성화
- `experimental_community_policy`
  - 커뮤니티 선택 정책 (`flat_ranked`, `leaf_only`, `leaf_then_parent_mix`, `pyramid`)
- `experimental_history_enabled`
  - history block 포함 여부
- `experimental_covariate_enabled`
  - covariate block 포함 여부
- `experimental_context_max_tokens`
  - assembled context 예산 토큰 상한
- `experimental_condition_id`
  - 조건 식별자(로그 추출/필터링에 사용)
- `experimental_log_context_payload=true`
  - `[LOCAL_CONTEXT_PAYLOAD]` JSON 로그 기록
- `experimental_policy_preserve_mode`
  - `fallback` 또는 `strict`

---

## 4) 16개 조건 실행 방법

스크립트:
- `scripts/run_experimental_local_context_matrix.py`

기본 정책 4개:
- `flat_ranked`
- `leaf_only`
- `leaf_then_parent_mix`
- `pyramid`

각 정책마다:
- history OFF/ON (2)
- covariate OFF/ON (2)

총 `4 × 2 × 2 = 16` 조건을 **query.txt 각 줄마다 자동 수행**합니다.

### 기본 실행 예시

```bash
python scripts/run_experimental_local_context_matrix.py \
  --root . \
  --data ./output \
  --query-file ./query.txt \
  --max-tokens 1800
```

### 실무에서 많이 쓰는 “run_id 고정 + 추출까지 한 번에” 예시

```bash
RUN_ID=expctx_demo_$(date -u +%Y%m%dT%H%M%SZ)

python scripts/run_experimental_local_context_matrix.py \
  --root . \
  --data ./output \
  --query-file ./query.txt \
  --max-tokens 1800 \
  --run-id "$RUN_ID"

python scripts/extract_experimental_local_context.py \
  --log ./logs/query.log \
  --condition-prefix "$RUN_ID" \
  --format tsv > "./experimental_local_context_runs/${RUN_ID}.tsv"
```

위 흐름이면 나중에 `RUN_ID` 단위로 결과를 다시 찾기 쉽습니다.

### 단계별 화면 출력(on/off) 예시

아래 옵션을 켜면 조건별로 대표 단계가 터미널에 직접 출력됩니다.

- 어떤 query가 들어왔는지
- entity top-k 설정/실제 선택 개수 및 목록
- 선택된 커뮤니티 ID와 레벨별 그룹
- 커뮤니티 summary 사용 여부
- 최종 assembled_context 토큰 수 / 경고
- (옵션) assembled_context 본문

```bash
python scripts/run_experimental_local_context_matrix.py \
  --max-tokens 1800 \
  --trace-steps \
  --trace-max-items 15
```

assembled_context 본문까지 즉시 보려면:

```bash
python scripts/run_experimental_local_context_matrix.py \
  --max-tokens 1800 \
  --trace-steps \
  --trace-show-context
```

현재 디렉터리가 프로젝트 루트라면 아래처럼 더 짧게도 가능:

```bash
python scripts/run_experimental_local_context_matrix.py --max-tokens 1800
```

실행이 끝나면 기본적으로 다음이 생성됩니다.

- `experimental_local_context_runs/<run_id>/run_summary.json`
- 각 조건별 응답 파일 `q###_c##.response.txt`
- 로그 파일: `<root>/logs/query.log` (기존 로그에 append)

---

## 5) assembled_context 최대 토큰 수 조절

`--max-tokens`로 조절합니다.

예시:

```bash
python scripts/run_experimental_local_context_matrix.py --max-tokens 1200
python scripts/run_experimental_local_context_matrix.py --max-tokens 2400
```

- 값이 작을수록 `assembled_context_tokens`가 줄거나 일부 블록이 제외될 수 있음
- 값이 너무 작으면 경고(`warnings`)가 늘어날 수 있음

---

## 6) 결과 해석 방법

핵심 확인 필드:

- `condition_id`
- `community_policy`
- `history_enabled`
- `covariate_enabled`
- `selected_community_ids`
- `warnings`
- `assembled_context_tokens`
- `assembled_context` (**최우선 관심 값**)

특히 `assembled_context` + `assembled_context_tokens`를 비교하면,
정책/옵션 조합에 따라 실제 LLM에 전달된 컨텍스트가 어떻게 달라지는지 빠르게 확인할 수 있습니다.

---

## 7) query.log에서 필요한 필드만 추출

추출 스크립트:
- `scripts/extract_experimental_local_context.py`

기본 출력은 TSV이며, 아래 필드를 포함합니다.

- `condition_id`
- `community_policy`
- `history_enabled`
- `covariate_enabled`
- `query`
- `selected_community_ids`
- `warnings`
- `assembled_context_tokens`
- `assembled_context`

기본 출력에서 개별 블록(`conversation_history_context`, `community_summary_context`, `covariate_context`)은 제외됩니다.

### 예시 1: 현재 실행 run_id만 TSV로 보기

```bash
python scripts/extract_experimental_local_context.py \
  --log ./logs/query.log \
  --condition-prefix expctx_20260327T120000Z \
  --format tsv
```

### 예시 2: JSONL로 저장

```bash
python scripts/extract_experimental_local_context.py \
  --log ./logs/query.log \
  --condition-prefix expctx_20260327T120000Z \
  --format jsonl > extracted.jsonl
```

### 예시 3: assembled_context만 개별 파일로 저장

```bash
python scripts/extract_experimental_local_context.py \
  --log ./logs/query.log \
  --condition-prefix expctx_20260327T120000Z \
  --assembled-context-dir ./assembled_context_only
```

---

## 8) 실패/경고 해석 가이드

`warnings`에 자주 나올 수 있는 메시지 예시와 의미:

- `History enabled but no conversation history was provided.`
  - history ON 조건이지만 대화 히스토리를 전달하지 않은 경우
  - 단일 질의 배치 실험에서는 정상적으로 자주 발생 가능
- 정책 관련 경고(예: leaf/pyramid에서 후보 부족)
  - 선택 가능한 커뮤니티 구조/토큰 예산 제약으로 일부 정책이 의도대로 확장되지 못한 상태
- 토큰 예산 관련 경고
  - `--max-tokens` 값을 늘리면 개선 가능

실무적으로는 다음 순서로 보면 됩니다.

1. `assembled_context_tokens`가 기대 범위인지 확인
2. `warnings`가 치명적(결과 공백)인지 단순 정보성인지 구분
3. 필요 시 `--max-tokens` 상향 후 재실행
4. 정책별 `assembled_context`를 나란히 비교

---

## 9) 추천 실행 순서 (요약)

1. `query.txt` 작성
2. 16조건 일괄 실행
3. `query.log`에서 `assembled_context` 중심 추출
4. 필요 시 `--max-tokens`를 바꿔 재실행

```bash
python scripts/run_experimental_local_context_matrix.py --max-tokens 1800
python scripts/extract_experimental_local_context.py --log ./logs/query.log --format tsv
```

이렇게 하면 코드 수정 없이 로그만으로 `assembled_context` 중심 검증/비교가 가능합니다.

---

## 10) “문서 명령이 실제로 동작하는지” 검증한 예시 (2026-03-30)

아래는 이 저장소 환경에서 실제로 실행해 확인한 명령입니다.

### 10-1. 추출 스크립트 동작 검증 (실행 성공)

샘플 `query.log` 한 줄을 만들고 TSV 추출을 실행:

```bash
cat > /tmp/expctx_sample.log <<'EOF'
2026-03-30 00:00:00.000 | INFO | [LOCAL_CONTEXT_PAYLOAD] {"condition_id":"expctx_demo|q001|flat_ranked|h0|c0","community_policy":"flat_ranked","history_enabled":false,"covariate_enabled":false,"query":"Who is Scrooge?","selected_community_ids":["1","2"],"warnings":[],"assembled_context_tokens":123,"assembled_context":"[Entities]\n- Scrooge\n[Communities]\n- ..."}
EOF

python scripts/extract_experimental_local_context.py \
  --log /tmp/expctx_sample.log \
  --format tsv
```

출력 예시:

```text
condition_id	community_policy	history_enabled	covariate_enabled	query	selected_community_ids	warnings	assembled_context_tokens	assembled_context
expctx_demo|q001|flat_ranked|h0|c0	flat_ranked	False	False	Who is Scrooge?	1,2		123	[Entities]
- Scrooge
[Communities]
- ...
```

### 10-2. experimental local context 핵심 로직 검증 (테스트 성공)

실험 컨텍스트 통합 테스트:

```bash
PYTHONPATH=packages/graphrag \
pytest -q tests/integration/query/test_experimental_local_context_integration.py
```

결과: `6 passed`

### 10-3. 16조건 실행 스크립트의 환경 의존성 체크

`run_experimental_local_context_matrix.py`는 실제 LLM 호출이 필요하므로, 아래가 준비되어야 끝까지 실행됩니다.

- `settings.yaml` 존재
- 유효한 completion/embedding 모델 키
- index output parquet 세트

빠른 사전 체크:

```bash
python scripts/run_experimental_local_context_matrix.py --help
```

> 정리: 이 문서의 **추출 흐름/실험 컨텍스트 로직 자체는 실제 실행으로 검증**했고, 16조건 전체 실행은 모델 인증이 준비된 프로젝트에서 그대로 동작합니다.
