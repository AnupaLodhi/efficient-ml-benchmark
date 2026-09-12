# Deployment image for the FastAPI inference service (deployment/api).
# CPU-only by design -- this project's RQ4 is specifically about
# CPU-deployment behavior, so we deliberately do NOT pull a CUDA base image
# even if one is available.
FROM python:3.11-slim

WORKDIR /app

# System deps for Pillow's image codecs.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch explicitly (avoids pulling multi-GB CUDA wheels
# that provide no benefit in this CPU-only deployment target).
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY deployment/requirements-api.txt /app/deployment/requirements-api.txt
RUN pip install --no-cache-dir -r /app/deployment/requirements-api.txt

# Only copy what the inference service actually needs at runtime -- not the
# whole repo (training code, notebooks, paper/, etc. have no business in
# the deployment image).
COPY src/models /app/src/models
COPY src/evaluation /app/src/evaluation
COPY src/utils /app/src/utils
COPY src/__init__.py /app/src/__init__.py
COPY configs /app/configs
COPY deployment/api /app/deployment/api
COPY deployment/__init__.py /app/deployment/__init__.py

# Model checkpoints and processed results are mounted at runtime (see
# docker-compose.yml) rather than baked into the image, so a new trained
# model doesn't require a rebuild.
RUN mkdir -p /app/models /app/results/processed

EXPOSE 8000
CMD ["uvicorn", "deployment.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
