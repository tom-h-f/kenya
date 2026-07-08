FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
# Persisted HF model cache (see compose volume) so ~3GB of transformer weights
# are downloaded once, not on every container restart.
ENV HF_HOME=/app/hf_cache

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY . .
RUN uv sync --frozen --no-dev

ENTRYPOINT ["uv", "run", "--no-sync", "kma-enrich"]
CMD ["--loop"]
