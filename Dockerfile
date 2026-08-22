# NextTrack API + demo page. Build from code/:  docker compose up
# The trained artifacts are baked into the image so the container serves with
# zero setup; the raw LFM-2b data (5.8 GB) is dockerignored and never enters
# the build context (and cannot be redistributed anyway).
#
# ARTIFACTS_DIR selects which model ships: the default 1M prototype artifacts
# (~25 MB) or the full-scale model (~420 MB):
#   docker build --build-arg ARTIFACTS_DIR=prototypes/artifacts_full -t ...:v2 .

FROM python:3.11-slim

ARG ARTIFACTS_DIR=prototypes/artifacts

# implicit's compiled ALS needs OpenMP, which slim images do not ship.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Dependency layer first (cached until the lockfile changes).
COPY pyproject.toml uv.lock README.md ./
COPY prototypes/nextrack prototypes/nextrack
RUN uv sync --frozen --no-dev

# Static demo page + trained model artifacts (see ARTIFACTS_DIR above).
COPY prototypes/web prototypes/web
COPY ${ARTIFACTS_DIR} prototypes/artifacts

ENV NEXTRACK_HOST=0.0.0.0
# The full-scale matrix-vector product benefits from BLAS threads at serving
# time (training pins to 1 thread itself, so this only affects inference).
# 2 matches the deploy target's vCPUs; override per host if different.
ENV OPENBLAS_NUM_THREADS=2
EXPOSE 8000

CMD ["uv", "run", "--no-sync", "api"]
