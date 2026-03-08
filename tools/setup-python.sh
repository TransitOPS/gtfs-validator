#!/usr/bin/env sh
set -eu

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Installing uv..."
  if command -v curl >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    PATH="$HOME/.local/bin:$PATH"
    export PATH
  else
    echo "curl is required to install uv automatically."
    echo "Install uv manually: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
  fi
fi

echo "Syncing Python dependencies..."
uv sync

echo "Setup complete. Run tests with: make test"
