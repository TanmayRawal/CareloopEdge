# Build from the repository root:  docker compose -f server/docker-compose.yml up -d --build
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 NE_DATA=/data
WORKDIR /app
COPY requirements-server.txt .
RUN pip install --no-cache-dir -r requirements-server.txt

# Only the modules the server needs (no training code, no model weights)
COPY neuroedge/__init__.py neuroedge/security.py neuroedge/bundle.py neuroedge/
COPY server/__init__.py server/app.py server/store.py server/manage.py server/

RUN useradd --uid 10001 --create-home neuroedge && mkdir -p /data && chown neuroedge /data
USER neuroedge
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "--factory", "server.app:create_app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
