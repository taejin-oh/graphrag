#!/usr/bin/env bash
set -euo pipefail

# Quick human-friendly verification runner for QFS scripts.
#
# Usage:
#   bash scripts/verify_qfs_quick.sh <TEST_CASE>
#
# Example:
#   bash scripts/verify_qfs_quick.sh case_a

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/verify_qfs_quick.sh <TEST_CASE>"
  exit 1
fi

TEST_CASE="$1"
RUN_ID="manual_check_$(date -u +%Y%m%dT%H%M%SZ)"

echo "[1/6] Branch check"
git branch --show-current

echo "[2/6] Script help check"
python scripts/run_qfs_index.py --help >/dev/null
python scripts/run_qfs_query_and_aggregate.py --help >/dev/null

echo "[3/6] Dry-run traversal check"
python scripts/run_qfs_index.py --dry-run --test-case "$TEST_CASE"
python scripts/run_qfs_query_and_aggregate.py --dry-run --test-case "$TEST_CASE" --run-id "$RUN_ID"

echo "[4/6] Index run"
python scripts/run_qfs_index.py --test-case "$TEST_CASE"

echo "[5/6] Query + aggregate run"
python scripts/run_qfs_query_and_aggregate.py --test-case "$TEST_CASE" --run-id "$RUN_ID"

echo "[6/6] Output sanity checks"
find "qfs_log/$RUN_ID" -maxdepth 2 -type f | sort

python - <<'PY'
import csv, glob, sys
run_root = sorted(glob.glob("qfs_log/manual_check_*"))
if not run_root:
    print("[WARN] No qfs_log/manual_check_* found")
    sys.exit(0)
latest = run_root[-1]
print(f"[INFO] latest run root: {latest}")
for path in sorted(glob.glob(f"{latest}/*/results.csv")):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    abst = [r for r in rows if r.get("question_type") == "abstention"]
    if abst:
        r = abst[0]
        print(path, "abstention=>", r.get("question"), r.get("selected_community_ids"), r.get("assembled_context_tokens"), r.get("assembled_context"))
PY

echo "Done. Run ID: $RUN_ID"
