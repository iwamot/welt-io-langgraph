#!/bin/bash
set -e

# mise
eval "$(mise activate bash)"
mise fmt
mise install

# Python
uv sync
uv run pip-licenses --partial-match --allow-only="Apache;BSD;CNRI-Python;ISC;MIT;MPL;PSF;Python Software Foundation"
uv audit
ruff check --fix
ruff format
ty check --error-on-warning
if [[ -n "$CI" ]]; then
  uv run pytest --cov --cov-report=term --cov-report=xml
else
  uv run pytest --cov --cov-report=term
fi
# README's Supported Versions table restates what pyproject.toml declares. Read
# both and compare, so an edit to one cannot leave the other behind.
uv run python - <<'PY'
import tomllib

from packaging.requirements import Requirement

with open("README.md", encoding="utf-8") as f:
    readme = f.read()
with open("pyproject.toml", "rb") as f:
    deps = tomllib.load(f)["project"]["dependencies"]
for dep in deps:
    req = Requirement(dep)
    spec = f"`{req.specifier}`" if req.specifier else "any"
    row = f"| `{req.name}` | {spec} |"
    if row not in readme:
        raise SystemExit(f"validate.sh: README.md has no row {row}")
    print(f"validate.sh: README.md states {req.name} {req.specifier or 'any'}")
PY

trap 'rm -rf dist' EXIT
rm -rf dist
uv build
# --token is a placeholder to skip the interactive prompt; --dry-run never uploads.
uv publish --dry-run --trusted-publishing never --token dry-run

# Shared lint tasks
mise run gha-lint
mise run shell-lint

# Check for uncommitted changes
git diff --exit-code
