# syntax=docker/dockerfile:1.7
# ForensiX server image (API + web manager).
#
#   docker build -t forensix:cpu .
#   docker build -t forensix:gpu --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu124 .
#   docker build -t forensix:cpu-bundled --build-arg MODELS="commfor dfdet-clip nes2net-x-itw syncnet c2pa watermark-image watermark-audio metadata" .
#
# MODELS bakes those models' weights into the image (no downloads at startup; runs with no
# network). Without it, weights download on first use into $FORENSIX_CACHE: mount a volume there.
# The GPU variant needs the NVIDIA driver and container toolkit on the host; PyTorch's CUDA
# wheels bring their own CUDA runtime.

ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 tini \
 && rm -rf /var/lib/apt/lists/*

# -- dependencies: a layer that only changes when pyproject.toml or the torch index does --------
FROM base AS deps
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN pip install torch torchvision --index-url "${TORCH_INDEX}"
WORKDIR /src
COPY pyproject.toml README.md ./
# a stub package, so pip resolves and installs the dependencies without the source
RUN mkdir forensix && touch forensix/__init__.py \
 && pip install ".[server]" \
 && pip uninstall -y forensix

# -- application -----------------------------------------------------------------------------
FROM deps AS app
COPY forensix ./forensix
RUN pip install --no-deps . && rm -rf /src
WORKDIR /app
RUN useradd --create-home --uid 10001 --user-group forensix \
 && mkdir -p /opt/forensix/cache /data /tmp/forensix \
 && chown -R forensix:forensix /opt/forensix /data /tmp/forensix
ENV FORENSIX_CACHE=/opt/forensix/cache \
    HOME=/home/forensix \
    XDG_CACHE_HOME=/tmp/forensix/xdg \
    MPLCONFIGDIR=/tmp/forensix/mpl \
    HF_HOME=/tmp/forensix/hf \
    HF_HUB_OFFLINE=1 \
    PORT=8000 \
    LOG_FORMAT=json \
    OMP_NUM_THREADS=4 \
    FORENSIX_SERVER_CALIBRATION_DIR=/data/calibration \
    FORENSIX_SERVER_GALLERY_DIR=/data/galleries \
    FORENSIX_SERVER_TMP_DIR=/tmp/forensix \
    FORENSIX_SERVER_BENCHMARKS_DIR=""
USER forensix

# -- optionally bake model weights in (verified, with hash stamps, so startup skips rehashing) --
ARG MODELS=""
RUN if [ -n "${MODELS}" ]; then \
      HF_HUB_OFFLINE=0 python -m forensix.server --warm-only --no-warmup --preload ${MODELS}; \
    fi

EXPOSE 8000
VOLUME ["/data"]
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
  CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8000\")}/health', timeout=4)" || exit 1
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-m", "forensix.server"]
