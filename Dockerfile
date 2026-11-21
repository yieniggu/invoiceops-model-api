# Build with the shared package as a named context (it lives outside this repo):
#   docker build --build-context ml=../invoiceops-ml -t invoiceops-model-api .
# uv.lock records the package as a path source "../invoiceops-ml", i.e. /invoiceops-ml here.
# Base images are pinned by tag AND digest of the multi-arch index; refresh them deliberately with
#   docker buildx imagetools inspect <image:tag>
FROM ghcr.io/astral-sh/uv:0.12.5@sha256:e85be844203885286c60ffad8a858d48afb6c5a5c237ca0e67f12e74b8f174b1 AS uv

FROM python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b AS builder

COPY --from=uv /uv /uvx /bin/
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

COPY --from=ml pyproject.toml README.md /invoiceops-ml/
COPY --from=ml src /invoiceops-ml/src
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN uv sync --frozen --no-dev
# The "ml" context is a path source: fail the build if its version is not the one uv.lock pins.
COPY scripts/check_ml_version.py /tmp/check_ml_version.py
RUN /app/.venv/bin/python /tmp/check_ml_version.py uv.lock

FROM python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b

RUN groupadd --system app && useradd --system --gid app --create-home app
WORKDIR /app
COPY --from=builder --chown=app:app /app/.venv .venv
COPY --from=builder --chown=app:app /app/src ./src

# Secrets (MODEL_API_SERVICE_TOKEN, OFFICIAL_VALIDATION_SEED, MLFLOW_TRACKING_TOKEN or
# username/password) are injected at runtime, never baked into the image.
ENV PATH="/app/.venv/bin:$PATH" \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MLFLOW_DISABLE_AGENT_HINT=1
USER app
EXPOSE 8000

# --no-access-log: the app writes its own structured audit lines; uvicorn access lines would add a
# second, unstructured format.
CMD ["uvicorn", "invoiceops_model_api.app:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
