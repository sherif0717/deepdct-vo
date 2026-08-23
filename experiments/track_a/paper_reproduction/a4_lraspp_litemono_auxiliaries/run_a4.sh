#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$HERE/preflight.sh"
"$HERE/train.sh"
"$HERE/evaluate.sh"
"$HERE/plot.sh"
