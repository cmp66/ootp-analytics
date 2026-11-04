#!/usr/bin/env bash
# On the server: pull the newest viewer files from S3 and restart the viewer only when something changed (it keeps the data in
# memory, so a restart is what shows new files). Run it by hand after `sync_files.py push --set viewer`, or from cron:
#   */15 * * * * /opt/ootp-viewer/deploy/refresh.sh >> /var/log/ootp-refresh.log 2>&1
set -euo pipefail
cd "$(dirname "$0")"
COMPOSE="docker compose -f docker-compose.viewer.yml"
out=$($COMPOSE exec -T viewer python scripts/sync_files.py pull --set viewer)
echo "$(date -u +%FT%TZ) $out"
if ! grep -q " 0 to download" <<<"$out"; then
    $COMPOSE restart viewer
fi
