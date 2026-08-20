# CPU-only image: guided capture UI + the "accurate" (COLMAP photogrammetry)
# pipeline, with a CPU meshing fallback when no GPU is present. This is
# enough to run the whole app end to end. For the "compelling" Gaussian
# splatting path (and full-quality dense multi-view stereo), see
# Dockerfile.gpu instead.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
      colmap \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY server/requirements.txt server/requirements.txt
RUN pip install --no-cache-dir -r server/requirements.txt

COPY server server
COPY web web

ENV DATA_DIR=/data
# No VOLUME instruction here - Railway rejects Dockerfiles that declare one
# and manages the /data mount entirely through its own Volumes feature
# instead (Settings -> Volumes -> mount path /data). Same effect, just
# configured on Railway's side rather than baked into the image.
EXPOSE 8000

# Shell form so ${PORT} expands - Railway (and similar PaaS) inject PORT at
# runtime and route their public domain to whatever port the app actually
# binds, rather than a fixed one. Falls back to 8000 for local `docker run`.
CMD uvicorn app.main:app --app-dir server --host 0.0.0.0 --port ${PORT:-8000}
