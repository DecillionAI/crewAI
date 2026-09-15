# syntax=docker/dockerfile:1.7
#
# The CrewAI **tool catalogue**, as one image, for a project's own machine.
#
# The other half of the engine/tools split. This image is what a space's Modal
# sandbox boots: the tool catalogue plus the tool server that exposes it to the
# platform over the gateway subscription channel. It runs no agents, holds no
# roster, and makes no model calls — its bridge grant names a fixed, short list
# of routes and the model endpoint is not among them, which is what makes a
# sandbox structurally unable to spend the platform's model budget.
#
# Preinstalling the catalogue here is the difference between a new project
# waiting minutes on `pip install crewai-tools` before anyone can list a folder,
# and one whose tools are ready as soon as the sandbox is.

FROM python:3.12-slim AS build

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential git pkg-config \
 && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/decillion/venv
ENV PATH="/opt/decillion/venv/bin:$PATH"

WORKDIR /src
COPY lib/crewai-core /src/lib/crewai-core
COPY lib/cli /src/lib/cli
COPY lib/crewai /src/lib/crewai
COPY lib/crewai-tools /src/lib/crewai-tools
COPY lib/tool-server /src/lib/tool-server

RUN pip install --upgrade pip setuptools wheel \
 && pip install /src/lib/crewai-core /src/lib/cli /src/lib/crewai \
 && pip install /src/lib/crewai-tools \
 && pip install /src/lib/tool-server

# ── runtime ─────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="crewai-sandbox" \
      org.opencontainers.image.description="CrewAI's tool catalogue plus the sandbox tool server." \
      org.opencontainers.image.source="https://github.com/cosmopole-org/crewai"

ENV PATH="/opt/decillion/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    XDG_DATA_HOME=/data/.crewai/data \
    XDG_CACHE_HOME=/data/.crewai/cache

# What a tool actually needs from the machine: a shell, a VCS, a fetcher, and
# the ability to unpack what it downloads. A tool that shells out to something
# missing fails in a way that reads as the tool being broken.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates git curl wget unzip zip tar procps jq ripgrep less \
 && rm -rf /var/lib/apt/lists/*

COPY --from=build /opt/decillion/venv /opt/decillion/venv
RUN ln -sf /opt/decillion/venv/bin/python /usr/local/bin/python

# The revision baked into this image. The sandbox's bootstrap compares the
# commit a fetch RESOLVED TO against this, never the ref's name — a marker
# reading "main" would match itself forever and the machine would run the first
# commit it ever saw for the rest of its life.
ARG RUNTIME_REF=""
RUN mkdir -p /opt/decillion && printf '%s\n' "${RUNTIME_REF}" > /opt/decillion/runtime-ref

WORKDIR /data
CMD ["decillion-tool-server"]
