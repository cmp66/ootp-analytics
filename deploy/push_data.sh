#!/usr/bin/env bash
# From your machine: send the processed viewer files to S3 and have the server pull them and restart the viewer (refresh.sh restarts
# it only when something changed). Run update_all.py first. For new code use deploy/deploy.sh.
#   deploy/push_data.sh [user@<lightsail-ip>]      (default $VIEWER_HOST; SSH_KEY=~/.ssh/lightsail.pem to name the key; PYTHON=...)
set -euo pipefail
HOST=${1:-${VIEWER_HOST:?usage: deploy/push_data.sh user@host (or set VIEWER_HOST)}}
SSH="ssh ${SSH_KEY:+-i $SSH_KEY}"
cd "$(dirname "$0")/.."
PYTHONPATH=. ${PYTHON:-python} scripts/sync_files.py push --set viewer
$SSH "$HOST" /opt/ootp-viewer/deploy/refresh.sh
