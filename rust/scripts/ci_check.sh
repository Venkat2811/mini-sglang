#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ -z "${VIRTUAL_ENV:-}" ]] && [[ -x "${ROOT_DIR}/../.venv/bin/python" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT_DIR}/../.venv/bin/activate"
fi

if command -v maturin >/dev/null 2>&1; then
  MATURIN_CMD=(maturin)
elif command -v uvx >/dev/null 2>&1; then
  MATURIN_CMD=(uvx maturin)
else
  echo "error: maturin or uvx is required" >&2
  exit 1
fi

cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
"${MATURIN_CMD[@]}" develop --manifest-path "${ROOT_DIR}/minisgl-cpu-py/Cargo.toml"
python -m unittest discover -s "${ROOT_DIR}/minisgl-cpu-py/tests" -p "test_*.py" -v
