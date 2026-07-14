#!/usr/bin/env bash
set -euo pipefail

FLOW_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$FLOW_ROOT/../.." && pwd)
CONFIG=${PPA_CONFIG:-"$FLOW_ROOT/config/default.env"}
OUTPUT=

while (($#)); do
  case "$1" in
    --output)
      [[ $# -ge 2 ]] || { echo "run_ppa.sh: --output requires a value" >&2; exit 2; }
      OUTPUT=$2
      shift 2
      ;;
    --help|-h)
      echo "usage: release/ppa-flow/run_ppa.sh --output <output-dir>"
      exit 0
      ;;
    *)
      echo "run_ppa.sh: unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$OUTPUT" ]]; then
  echo "run_ppa.sh: --output is required" >&2
  exit 2
fi

set -a
# shellcheck disable=SC1090
source "$CONFIG"
set +a

FILELIST=${PPA_FILELIST:-rtl/files.f}
if [[ "$FILELIST" != /* ]]; then
  FILELIST="$REPO_ROOT/$FILELIST"
fi

exec python3 "$FLOW_ROOT/run_ppa.py" \
  --output "$OUTPUT" \
  --filelist "$FILELIST" \
  --top "${PPA_TOP:-aec_eval_top}"
