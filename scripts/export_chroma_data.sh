#!/usr/bin/env bash
# Archive the active Chroma 1.5 persistence directory for an EC2 seed restore.
set -euo pipefail

container_name="${CHROMA_CONTAINER:-chroma}"
archive_path="${1:?usage: scripts/export_chroma_data.sh <output.tar>}"

mkdir -p "$(dirname "$archive_path")"
docker exec "$container_name" tar -C /data -cf - . > "$archive_path"
shasum -a 256 "$archive_path" | tee "${archive_path}.sha256"
