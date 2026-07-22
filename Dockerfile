# syntax=docker/dockerfile:1

FROM node:22-alpine AS frontend-builder

WORKDIR /build/frontend

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./

# Vite replaces this public URL at build time. No secret is included in the
# browser bundle; the OpenAI key is supplied only to the runtime container.
ARG VITE_API_BASE_URL=http://localhost:5173
ENV VITE_API_BASE_URL=${VITE_API_BASE_URL}
RUN npm run build


FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install --no-install-recommends -y nginx \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 appuser

WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY agent_layer/ agent_layer/
COPY api_layer/ api_layer/
COPY mcp_server/ mcp_server/
COPY scripts/start_container.py scripts/start_container.py
COPY deploy/nginx.conf /etc/nginx/nginx.conf
COPY --from=frontend-builder /build/frontend/dist/ /usr/share/nginx/html/

RUN mkdir -p \
        /tmp/nginx-client-body \
        /tmp/nginx-proxy \
        /tmp/nginx-fastcgi \
        /tmp/nginx-uwsgi \
        /tmp/nginx-scgi \
    && chown -R appuser:appuser /tmp/nginx-*

USER appuser

EXPOSE 5173

HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=5 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5173/api/health', timeout=2).read()"]

CMD ["python", "-u", "scripts/start_container.py"]
