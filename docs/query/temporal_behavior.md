# Temporal behavior in community summarization and query processing

This document explains how temporal information is currently applied in GraphRAG for:

1. community summarization (indexing phase)
2. local/QFS query context building (query phase)

It is intentionally short and practical.

## 1) Community summarization (indexing)

### 1.1 Temporal metadata is created and preserved

Text units store temporal fields such as:

- `start_turn_index`, `end_turn_index`
- `turn_timestamp_start`, `turn_timestamp_end`
- `chunk_index_in_conversation`

These fields are copied into the local community context (`ALL_CONTEXT`) used for report generation.

### 1.2 Sorting policy for community source context

`sort_context(...)` uses the following order:

1. **Importance first**: higher `entity_degree`
2. **Temporal tie-breaker** (only when importance is equal):
   - `start_turn_index`
   - `turn_timestamp_start`
   - `chunk_index_in_conversation`
3. deterministic final key: `id`

So the policy is **not** “latest always wins”.
Temporal order is used when needed, without overriding primary importance.

### 1.3 Community report output is temporally structured

The report prompt/schema includes:

- `current_state`
- `timeline_events`
- `superseded_facts`
- `date_range`

This enables a summary that keeps overall key points while explicitly representing time evolution and superseded facts.

---

## 2) Local / QFS query processing

### 2.1 Community context in experimental mode is summary-only

For experimental assembled context, community input is built with `use_community_summary=True`.
This means community **summary** is used, not raw full content.

### 2.2 Text-unit temporal handling during query

For selected text units, local query context builder:

- sorts units by temporal metadata
- splits them into:
  - `CURRENT` (max `end_turn_index`)
  - `TIMELINE` (older units)

This gives the model both latest state and history.

### 2.3 Response instruction is temporal-aware

The local search system prompt asks the model to structure answers as:

1. CURRENT STATE
2. TIMELINE / HISTORY
3. SUPERSEDED FACTS

So query answering is guided to separate current facts from older/superseded ones.

---

## 3) Practical interpretation

Current behavior is best described as:

- **Importance-driven summarization** with **temporal support**
- not a strict recency-only system

In short:

- community summarization prioritizes key evidence
- temporal fields are used to organize/clarify evolution when needed
- query flow keeps summary-only community context and temporal answer structure
