#!/usr/bin/env bash
# Make this package's tests runnable on any host.
#
# The bridge declares crewai, which pins Python to <3.14, so `pip install -e .`
# refuses on a newer interpreter — and the suite was then reported as
# unrunnable, even though none of its tests import crewai (every crewai import
# in the package is lazy, inside the function that needs it). This installs only
# what the tests actually need and leaves the package on `pythonpath`, which
# pyproject.toml already sets for pytest.
#
#   scripts/dev-setup.sh && .venv/bin/python -m pytest
#
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"
"$PYTHON" -m venv .venv
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet pytest pytest-asyncio "websockets>=13,<16"

# crewai only installs on a supported interpreter. Where it does, install it, so
# the roster/tool builders can be exercised for real rather than skipped.
if .venv/bin/python -c 'import sys; sys.exit(0 if sys.version_info < (3, 14) else 1)'; then
  .venv/bin/pip install --quiet crewai || echo "note: crewai did not install; tests that need it will be skipped"
else
  echo "note: $($PYTHON -V) is outside crewai's supported range; running the suite without it"
fi

.venv/bin/python -m pytest
