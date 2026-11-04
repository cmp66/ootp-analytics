# The viewer only (no training libraries). Built on the server by deploy/docker-compose.viewer.yml.
FROM python:3.14-slim

WORKDIR /app
COPY deploy/requirements-viewer.txt /tmp/requirements-viewer.txt
RUN pip install --no-cache-dir -r /tmp/requirements-viewer.txt && rm /tmp/requirements-viewer.txt

COPY config ./config
COPY model ./model
COPY stats ./stats
COPY util ./util
COPY viewer ./viewer
COPY scripts ./scripts

ENV PYTHONPATH=/app PYTHONUNBUFFERED=1
# the league files are a volume (scripts/sync_files.py pull --set viewer fills it)
VOLUME /app/files
EXPOSE 8501
CMD ["streamlit", "run", "viewer/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
