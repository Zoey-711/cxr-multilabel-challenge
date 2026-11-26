#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIC100=""
PIC5=""
OUTPUT="$ROOT/reports"
INDEX=""
ONLY=""
OFFICIAL=0
declare -A SUBMISSIONS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --pictures-100) PIC100="$2"; shift 2 ;;
    --pictures-5) PIC5="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --image-index) INDEX="$2"; shift 2 ;;
    --candidate) ONLY="$2"; shift 2 ;;
    --submission)
      [[ "$2" == *=* ]] || { echo "Use --submission CANDIDATE=/path/to/submission.csv" >&2; exit 2; }
      SUBMISSIONS["${2%%=*}"]="${2#*=}"; shift 2 ;;
    --official-sample) OFFICIAL=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$PIC100" && -n "$PIC5" ]] || { echo "Both image directories are required" >&2; exit 2; }
python "$ROOT/verify_packages.py"
python "$ROOT/check_environment.py" --output "$OUTPUT/environment.json"
FAILURES=0
MATCHED=0
for NAME in E08_ONNX_FP32 H17_BATCH1_FP16 E08_PYTORCH_FP16 H17_ONNX_FP32; do
  [[ -z "$ONLY" || "$ONLY" == "$NAME" ]] || continue
  MATCHED=$((MATCHED + 1))
  CAND="$ROOT/candidates/$NAME"
  ARGS=(--pictures-100 "$PIC100" --pictures-5 "$PIC5" --output "$OUTPUT/$NAME")
  if [[ -n "${SUBMISSIONS[$NAME]:-}" ]]; then
    ARGS+=(--submission "${SUBMISSIONS[$NAME]}")
  elif [[ -f "$CAND/reference_submission.csv" ]]; then
    ARGS+=(--submission "$CAND/reference_submission.csv")
  fi
  if [[ -n "$INDEX" ]]; then ARGS+=(--image-index "$INDEX"); fi
  if [[ "$OFFICIAL" -eq 1 ]]; then ARGS+=(--official-sample); fi
  if python "$CAND/cost.py" "${ARGS[@]}"; then
    echo "$NAME: completed"
  else
    echo "$NAME: FAILED; other candidates continue" >&2
    FAILURES=$((FAILURES + 1))
  fi
done
[[ "$MATCHED" -gt 0 ]] || { echo "Unknown candidate: $ONLY" >&2; exit 2; }
[[ "$FAILURES" -eq 0 ]] || exit 1
