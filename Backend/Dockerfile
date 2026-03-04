# Build from repo ROOT context: docker build -f Backend/Dockerfile -t codeforge-backend .

FROM python:3.11-slim AS builder
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Runtime image ─────────────────────────────────────────────────────
FROM python:3.11-slim
WORKDIR /app

# Install Node.js 20 LTS (required by main.py to run pptxgenjs for PPTX export)
RUN apt-get update && apt-get install -y --no-install-recommends \
      curl ca-certificates gnupg \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && apt-get purge -y curl gnupg \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy Backend source (package.json must exist here for pptxgenjs)
COPY Backend/ .

# Install pptxgenjs (and any other Node deps in Backend/package.json)
RUN npm install --omit=dev

# Security: run as non-root
RUN adduser --disabled-password --gecos "" appuser && \
    chown -R appuser:appuser /app
USER appuser

ENV PORT=4001
EXPOSE 4001
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "4001"]