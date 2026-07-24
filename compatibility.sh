#!/bin/bash
set -euo pipefail

# mise (uv only — a full `mise install` would put the pinned Python ahead of
# the version under test)
eval "$(mise activate bash)"
mise install aqua:astral-sh/uv

# CI (compatibility.yml) runs this once per supported Python version — the
# requires-python floor upward — by setting UV_PYTHON.
# Locally: UV_PYTHON=3.10 ./compatibility.sh
export UV_PROJECT_ENVIRONMENT=".venv-compat/${UV_PYTHON:-default}"

uv sync
# Tests import all of src, but nothing imports the example agent, so
# byte-compile it to catch syntax incompatibilities there.
uv run --no-sync python -m compileall -q examples
uv run --no-sync pytest
