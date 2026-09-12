# Deployment

A FastAPI inference backend + Streamlit frontend, serving whichever trained
checkpoints actually exist under `models/` (baseline and/or compressed
variants). Nothing here fabricates a model or a metric that wasn't produced
by Phases 2-6 -- see `deployment/api/main.py` and `model_registry.py`
docstrings.

## Status

**Code-complete, tested with `TestClient`/`uvicorn`/`streamlit run` against
a freshly-initialized (untrained) checkpoint. Not yet run against real
trained models**, since Phases 2-5 haven't produced any (see
`PHASE_LOG.md`). The API correctly reports "no models available" / `None`
for unmeasured metrics rather than inventing numbers -- verified by
`tests/test_deployment_api.py`.

**Docker images are written but UNTESTED in this sandbox** -- this
container has no Docker daemon and no access to Docker Hub (network
egress is restricted to a small package-registry allowlist; see
`README.md` root for the full list). Validate `docker compose build` on a
machine with Docker before relying on it.

## Running locally (no Docker)

```bash
# Terminal 1 -- API
pip install -r requirements.txt   # or deployment/requirements-api.txt for a leaner install
uvicorn deployment.api.main:app --reload --port 8000

# Terminal 2 -- frontend
streamlit run deployment/frontend/app.py
```

Then train at least the baseline (`python -m experiments.train_baseline`)
so `models/baseline/*.pt` exists -- the API will report "no models
available" (by design, not a bug) until it does.

## Running with Docker

```bash
docker compose up --build
# API:      http://localhost:8000
# Frontend: http://localhost:8501
```

`docker-compose.yml` mounts `models/`, `results/processed/`, and
`configs/` read-only into the API container, so newly trained checkpoints
appear after a container restart without rebuilding the image.

## Where this actually runs (the three options discussed)

1. **Local machine** -- `docker compose up`, simplest, zero cost, good
   enough to generate the paper's "Deployment Evaluation" numbers if your
   machine is the reference CPU.
2. **A small VPS** (e.g. a $5-10/mo droplet) -- same image, `docker compose
   up -d` there instead. Most paper-appropriate option: gives a real,
   reproducible CPU spec to report latency numbers against, and a public
   URL if you want a live demo link.
3. **A managed platform** (Render/Fly.io/Railway/HF Spaces) -- least setup,
   but you don't control exact CPU specs, which matters if precise latency
   numbers are going in the paper.

Recommended path: validate locally first, then redeploy the identical
image to a VPS once you want the real deployment-evaluation numbers.

## API Reference

- `GET /health` -- liveness check.
- `GET /models` -- lists every checkpoint found under `models/{baseline,
  pruning,quantization,combined}/*.pt`, joined against
  `results/processed/results_processed.csv` for measured metrics where
  available.
- `POST /predict?model_id=<id>` (multipart file upload) -- runs inference,
  returns predicted class, confidence, **this request's actual latency**,
  and (separately) the **benchmarked** latency/size/compression ratio from
  Phase 6, or `null` if that model hasn't been benchmarked yet. The two
  latency numbers are never conflated.
