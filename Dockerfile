
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/gitcontext/.cache/huggingface

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system gitcontext \
    && useradd --system --gid gitcontext --create-home gitcontext \
    && mkdir -p /app/data/repos /app/data/qdrant "$HF_HOME" \
    && chown -R gitcontext:gitcontext /app /home/gitcontext

COPY requirements-docker.txt .

RUN pip install --no-cache-dir --disable-pip-version-check --no-deps \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.14.1+cpu \
    && pip install --no-cache-dir --disable-pip-version-check \
        -r requirements-docker.txt

COPY --chown=gitcontext:gitcontext src ./src

USER gitcontext

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=5 \
    CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:8000/health', timeout=4)"

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
