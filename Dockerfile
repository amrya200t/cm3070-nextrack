# NextTrack API + demo page. Build from code/:  docker compose up
# The trained artifacts (~25 MB) are baked into the image so the container
# serves with zero setup; the raw LFM-2b data (5.8 GB) is dockerignored and
# never enters the build context (and cannot be redistributed anyway).

FROM python:3.11-slim

# implicit's compiled ALS needs OpenMP, which slim images do not ship.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Dependency layer first (cached until the lockfile changes).
COPY pyproject.toml uv.lock README.md ./
COPY prototypes/nextrack prototypes/nextrack
RUN uv sync --frozen --no-dev

# Static demo page + trained model artifacts.
COPY prototypes/web prototypes/web
COPY prototypes/artifacts prototypes/artifacts

ENV NEXTRACK_HOST=0.0.0.0
EXPOSE 8000

CMD ["uv", "run", "--no-sync", "api"]
