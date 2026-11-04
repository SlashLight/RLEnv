FROM python:3.11-slim

RUN pip install --no-cache-dir requests==2.32.3 \
 && apt-get update \
 && apt-get install -y --no-install-recommends curl jq procps ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# The container runs as the invoking host uid; no privileged user is needed.
USER 1000:1000
WORKDIR /workspace
