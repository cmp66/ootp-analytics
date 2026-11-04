#!/usr/bin/env bash
# From your machine: copy the code to the Lightsail host and (re)build the viewer container. The league files are not copied; they
# come from S3 (deploy/refresh.sh). Nothing outside /opt/ootp-viewer and the viewer container is touched.
#   deploy/deploy.sh ubuntu@<lightsail-ip>            (SSH_KEY=~/.ssh/lightsail.pem to name the key)
set -euo pipefail
HOST=${1:?usage: deploy/deploy.sh user@host}
REMOTE=/opt/ootp-viewer
SSH="ssh ${SSH_KEY:+-i $SSH_KEY}"
cd "$(dirname "$0")/.."
$SSH "$HOST" "sudo mkdir -p $REMOTE/files && sudo chown -R \$USER $REMOTE"
rsync -az --delete -e "$SSH" --exclude '__pycache__' --exclude 'deploy/.env' --exclude '*.pyc' \
    config model stats util viewer scripts deploy "$HOST:$REMOTE/"
$SSH "$HOST" "cd $REMOTE && docker compose -f deploy/docker-compose.viewer.yml up -d --build && docker image prune -f"
