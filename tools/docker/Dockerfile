# trochilus-dev: build, tests and oracles on Linux, independent of Windows' Smart App Control
# (docs/LESSONS.md #12). gcc + clang, and the Python reference (torch CPU, transformers, gguf)
# with the same pins as tools/requirements-oracle.txt.
#   docker build -t trochilus-dev:local -f tools/docker/Dockerfile tools
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive PYTHONUTF8=1
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential clang python3 python3-venv ca-certificates git \
    && rm -rf /var/lib/apt/lists/*
COPY requirements-oracle.txt /opt/requirements-oracle.txt
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir -r /opt/requirements-oracle.txt
ENV PY=/opt/venv/bin/python
RUN mkdir -p /work && chown ubuntu:ubuntu /work
WORKDIR /src
