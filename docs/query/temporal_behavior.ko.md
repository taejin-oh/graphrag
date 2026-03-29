# 커뮤니티 요약 및 QFS/Local Query의 시간 정보 적용 방식

이 문서는 GraphRAG에서 시간 정보가 현재 어떻게 적용되는지,
아래 두 흐름으로 나누어 간단히 설명합니다.

1. 커뮤니티 요약(인덱싱 단계)
2. Local/QFS Query 컨텍스트 구성(질의 단계)

## 1) 커뮤니티 요약(인덱싱)

### 1.1 시간 메타데이터 생성 및 보존

텍스트 유닛에는 다음 시간 필드가 저장됩니다.

- `start_turn_index`, `end_turn_index`
- `turn_timestamp_start`, `turn_timestamp_end`
- `chunk_index_in_conversation`

이 값들은 커뮤니티 보고서 생성을 위한 로컬 컨텍스트(`ALL_CONTEXT`)에도 전달됩니다.

### 1.2 커뮤니티 소스 정렬 정책

`sort_context(...)` 정렬 우선순위는 다음과 같습니다.

1. **중요도 우선**: `entity_degree` 높은 순
2. **시간 정보는 동점 보조키**:
   - `start_turn_index`
   - `turn_timestamp_start`
   - `chunk_index_in_conversation`
3. 마지막 결정성 키: `id`

즉, 정책은 **최신 정보 무조건 우선**이 아닙니다.
시간은 필요할 때(중요도 동률 등) 보조적으로 사용됩니다.

### 1.3 커뮤니티 리포트의 시간 구조

커뮤니티 리포트 프롬프트/스키마에는 아래 필드가 포함됩니다.

- `current_state`
- `timeline_events`
- `superseded_facts`
- `date_range`

그래서 전반적인 핵심 요약을 유지하면서,
시간 흐름과 이미 대체된 사실을 함께 표현할 수 있습니다.

---

## 2) Local / QFS Query 처리

### 2.1 experimental 모드에서 커뮤니티 컨텍스트는 summary-only

assembled context 경로에서 커뮤니티 데이터는 `use_community_summary=True`로 구성됩니다.
즉 커뮤니티 **원문(full content)**이 아니라 **요약(summary)**을 사용합니다.

### 2.2 질의 시 텍스트 유닛의 시간 처리

선정된 텍스트 유닛에 대해 로컬 질의 컨텍스트 빌더는:

- 시간 메타데이터 기준 정렬을 수행하고,
- 다음처럼 분리합니다.
  - `CURRENT`: `end_turn_index`가 최대인 유닛
  - `TIMELINE`: 그 이전 이력 유닛

이로써 최신 상태와 과거 흐름을 동시에 제공할 수 있습니다.

### 2.3 응답 프롬프트의 시간 인지 지시

로컬 검색 시스템 프롬프트는 답변을 다음 구조로 작성하도록 유도합니다.

1. CURRENT STATE
2. TIMELINE / HISTORY
3. SUPERSEDED FACTS

즉, 응답 단계에서도 현재/이력/대체 사실을 분리해 설명하도록 설계되어 있습니다.

---

## 3) 실무적으로 해석하면

현재 동작은 다음으로 요약할 수 있습니다.

- **중요도 중심 요약 + 시간 정보 보조 활용**
- **recency-only(최신 일변도) 시스템은 아님**

정리하면:

- 커뮤니티 요약은 핵심성(중요도)을 우선하고,
- 시간 정보는 흐름/대체관계가 필요할 때 사용되며,
- 질의 단계는 summary-only 커뮤니티 컨텍스트와 시간 구조화된 응답 지시를 사용합니다.
