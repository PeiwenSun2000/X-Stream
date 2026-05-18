#!/usr/bin/env bash
# Build the two micro JSONLs used by the QA / smoke tests.
#
#   sample_10_merged.jsonl  - first 10 entries of eval_relative_merged_phostream_type.jsonl
#                             (single-stream merged videos; pairs --multi-stream pixel)
#   sample_10_multi.jsonl   - first 10 entries of eval_relative_multi_phostream_type.jsonl
#                             (two video streams per round; pairs --multi-stream
#                             time/code/code_adaptive/cdpruner/surge)
#
# These outputs are git-ignored. Re-run this script any time to regenerate them.

set -euo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
N="${1:-10}"
DATA_DIR="${X_STREAM_DATA_DIR:-${THIS_DIR}/../../data/v1}"

src_merged="${DATA_DIR}/eval_relative_merged_phostream_type.jsonl"
src_multi="${DATA_DIR}/eval_relative_multi_phostream_type.jsonl"

[ -f "${src_merged}" ] || { echo "Error: missing ${src_merged}" >&2; exit 1; }
[ -f "${src_multi}" ]  || { echo "Error: missing ${src_multi}"  >&2; exit 1; }

head -n "${N}" "${src_merged}" > "${THIS_DIR}/sample_${N}_merged.jsonl"
head -n "${N}" "${src_multi}"  > "${THIS_DIR}/sample_${N}_multi.jsonl"

echo "Wrote:"
echo "  ${THIS_DIR}/sample_${N}_merged.jsonl  ($(wc -l < "${THIS_DIR}/sample_${N}_merged.jsonl") lines)"
echo "  ${THIS_DIR}/sample_${N}_multi.jsonl   ($(wc -l < "${THIS_DIR}/sample_${N}_multi.jsonl") lines)"
