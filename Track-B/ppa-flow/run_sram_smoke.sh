#!/usr/bin/env bash
set -euo pipefail

FLOW_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CONFIG=${PPA_CONFIG:-"$FLOW_ROOT/config/default.env"}

set -a
# shellcheck disable=SC1090
source "$CONFIG"
set +a

if [[ -z ${ASAP7_SRAM_ROOT:-} ]]; then
  echo "run_sram_smoke.sh: ASAP7_SRAM_ROOT is required" >&2
  exit 2
fi

OUTPUT=${1:-"$FLOW_ROOT/build/sram-smoke"}
exec python3 "$FLOW_ROOT/run_ppa.py" \
  --output "$OUTPUT" \
  --filelist "$FLOW_ROOT/smoke/sram-files.f" \
  --top sram_smoke_top
