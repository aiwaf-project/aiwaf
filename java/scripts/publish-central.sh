#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

require_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required env var: $name" >&2
    exit 1
  fi
}

require_env CENTRAL_TOKEN_USERNAME
require_env CENTRAL_TOKEN_PASSWORD
require_env GPG_PASSPHRASE

if ! command -v gpg >/dev/null 2>&1; then
  echo "gpg not found in PATH" >&2
  exit 1
fi

echo "Running tests before publish..."
mvn -q test

echo "Publishing to Maven Central via Sonatype Central plugin..."
mvn -s settings-central-template.xml -DskipTests deploy

echo "Publish command completed."
