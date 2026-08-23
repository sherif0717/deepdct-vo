#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$HERE/preflight.sh"
bash "$HERE/evaluate_seq09.sh"
bash "$HERE/evaluate_seq10.sh"

echo "A5 evaluations complete."
echo "Use plot_trajectory.py to generate GT-vs-pred trajectory figures."
