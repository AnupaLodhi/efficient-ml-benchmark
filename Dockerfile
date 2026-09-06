# Placeholder — built out properly in Phase 7 (Deployment).
# This will containerize the FastAPI inference service in deployment/api,
# using a CPU-only PyTorch base to keep the image small and match the
# CPU-deployment focus of RQ4.
FROM python:3.11-slim

WORKDIR /app

# TODO(Phase 7): copy only what the inference service needs (src/models,
# src/utils, deployment/api, the chosen model checkpoint), install a
# CPU-only torch build, and set the FastAPI entrypoint via uvicorn.

CMD ["echo", "Dockerfile not yet implemented -- see Phase 7 in PHASE_LOG.md"]
