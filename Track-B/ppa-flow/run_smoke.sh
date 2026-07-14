#!/usr/bin/env bash
set -euo pipefail

FLOW_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CONFIG=${PPA_CONFIG:-"$FLOW_ROOT/config/default.env"}

set -a
# shellcheck disable=SC1090
source "$CONFIG"
set +a

OUTPUT=${1:-"$FLOW_ROOT/build/smoke"}
exec python3 "$FLOW_ROOT/run_ppa.py" \
  --output "$OUTPUT" \
  --filelist "$FLOW_ROOT/smoke/files.f" \
  --top aec_eval_top
