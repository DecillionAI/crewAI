# syntax=docker/dockerfile:1.7
#
# The CrewAI **agentic engine**, as one image.
#
# This is half of the split the deployment makes: the engine plans, delegates
# and calls models, and runs on the Caspar node as a docker creature. It has no
# tools in it, on purpose — a tool touches a project's files, its shell and its
# network, and that belongs on the project's own machine, not on the node that
# runs everybody's agents. The other half is `sandbox.Dockerfile`.
#
# Nothing Caspar-specific is baked in here. The creature that adapts this to the
# node's bridge protocol layers on top of this image (autobot's
# `creatures/crewai/Dockerfile`), so this one stays a plain, runnable CrewAI.

FROM python:3.12-slim AS build

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential git \
 && rm -rf /var/lib/apt/lists/*

# A venv rather than the system interpreter: it copies to the runtime stage as
# one self-contained directory, so the compiler and headers above never ship.
RUN python -m venv /opt/crewai
ENV PATH="/opt/crewai/bin:$PATH"

WORKDIR /src
# The three workspace packages the engine is. They pin each other by exact
# version (all 1.15.21), so installing the local directories together satisfies
# those pins from the checkout instead of resolving them from the index — which
# is what makes this image build the code in THIS commit rather than whatever
# the last release happened to be.
COPY lib/crewai-core /src/lib/crewai-core
COPY lib/cli /src/lib/cli
COPY lib/crewai /src/lib/crewai
RUN pip install --upgrade pip setuptools wheel \
 && pip install /src/lib/crewai-core /src/lib/cli /src/lib/crewai

# ── runtime ─────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="crewai-engine" \
      org.opencontainers.image.description="The CrewAI agentic engine, without the tool catalogue." \
      org.opencontainers.image.source="https://github.com/cosmopole-org/crewai"

ENV PATH="/opt/crewai/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CREWAI_HOME=/opt/crewai \
    # CrewAI writes memory, telemetry and cache under the user data dir. In a
    # container that is a path that may not exist and may not be writable; give
    # it one that does and is.
    XDG_DATA_HOME=/var/lib/crewai \
    XDG_CACHE_HOME=/var/cache/crewai

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates git curl \
 && rm -rf /var/lib/apt/lists/* \
 && mkdir -p /var/lib/crewai /var/cache/crewai

COPY --from=build /opt/crewai /opt/crewai

# There is deliberately NO symlink from /usr/local/bin/python into this venv.
#
# `python -m venv` records the interpreter AS INVOKED, so building the venv with
# `python` makes /opt/crewai/bin/python a link to /usr/local/bin/python. Pointing
# /usr/local/bin/python back at the venv therefore closes a two-link cycle, and
# every `python` in the image — including the venv's own — then fails with
# "Too many levels of symbolic links". The venv works precisely BECAUSE that
# base link still resolves to the real interpreter.
#
# Nothing is lost by leaving it alone: this base image already ships
# /usr/local/bin/python, and PATH above puts the venv's first, so `python` in
# any process that inherits the image's environment is already this venv's.
# A LOGIN shell rebuilds PATH from /etc/profile and would miss it, which is what
# this covers instead.
RUN printf 'PATH="/opt/crewai/bin:$PATH"\n' > /etc/profile.d/10-crewai-venv.sh

WORKDIR /app
RUN python -c "import crewai; print('crewai', crewai.__version__)"

CMD ["python", "-c", "import crewai; print('crewai', crewai.__version__, 'engine image — layer a runtime on top of this')"]
