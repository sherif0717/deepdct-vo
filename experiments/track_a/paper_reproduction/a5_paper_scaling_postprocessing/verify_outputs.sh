#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
status=0
for seq in 09 10; do
  for mode in unscaled scaled; do
    d="$HERE/sequence_${seq}/${mode}"
    if [[ ! -d "$d" ]]; then
      echo "[MISSING] $d"
      status=1
    else
      echo "[OK] $d"
    fi
  done
done
exit "$status"
