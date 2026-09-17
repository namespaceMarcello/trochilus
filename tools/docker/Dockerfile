# trochilus-dev: build, test e oracoli su Linux, indipendenti da Smart App Control di Windows
# (docs/LEZIONI.md #12). gcc + clang, e il riferimento Python (torch CPU, transformers, gguf)
# con gli stessi pin di tools/requirements-oracle.txt.
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
