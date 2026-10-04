#!/usr/bin/env bash
# Restore an archive made by export_chroma_data.sh into an empty EC2 Chroma volume.
set -euo pipefail

archive_path="${1:?usage: scripts/restore_chroma_data.sh <archive.tar>}"
volume_name="${CHROMA_VOLUME:-298p_chroma-data}"
archive_file="$(basename "$archive_path")"

test -f "$archive_path"
test -n "$(docker volume ls --quiet --filter "name=^${volume_name}$")" || {
  echo "Docker volume '$volume_name' does not exist. Start the Chroma service once, then stop it." >&2
  exit 1
}
docker run --rm \
  -v "${volume_name}:/data" \
  -v "$(cd "$(dirname "$archive_path")" && pwd):/backup:ro" \
  -e ARCHIVE_FILE="$archive_file" \
  alpine sh -c 'rm -rf /data/* /data/.[!.]* /data/..?*; tar -C /data -xf "/backup/$ARCHIVE_FILE"'
