# decillion-tool-server

A Decillion project's sandbox tool server. Agents and their orchestration run
on Caspar; this process only exposes the project's filesystem and other
sandbox-local tools to them.

One process runs per project as the sandbox entrypoint:

```text
Caspar agent -- tool call --> bridge --> sandbox tool
tool result ---------------- gateway --> Caspar agent
```

## Authentication

The bridge holds no Caspar private key. It presents a bearer token minted by
the platform for one project topic (`space:<id>`). The node binds that grant to
the exact Caspar programs the bridge may reach. A sandbox cannot address an
arbitrary creature or invoke `llm/chat`.

## Configuration

The platform writes `/etc/decillion/bridge.env` when it provisions the sandbox:

| Variable | Meaning |
|----------|---------|
| `CASPAR_GATEWAY_URL` | WebSocket URL of the Caspar node |
| `DECILLION_SPACE_ID` | Project served by this process |
| `DECILLION_BRIDGE_TOPIC` | Gateway topic (defaults to `space:<id>`) |
| `DECILLION_BRIDGE_TOKEN` | Bearer token; never logged or sent back |
| `CREWAI_HOME` | Location of the CrewAI checkout |
| `DECILLION_LOG_LEVEL` | Log level (default `INFO`) |
| `DECILLION_STATE_DIR` | Durable state on the project's own volume |
| `DECILLION_RUNTIME_REF` | Installed tool-catalogue revision |

## Tool calls

Caspar sends a call from `crew/tool` through `crew/bridge`. The bridge executes
the named CrewAI sandbox tool and returns its result through the gateway. It
does not run an agent, call a model, or hold a model-provider key.

A sandbox sleeps when idle and the bridge may restart. The outbox therefore
writes every result to the project's volume before sending it, retries delivery
with backoff, and replays anything a dead process left behind.

On connect, the server announces its installed revision and tool catalogue.
Caspar can then release calls queued while the sandbox was starting or asleep.
Work activity and history remain authoritative in Caspar's run ledger; the
sandbox has no second agent lifecycle to reconcile with it.

## Tests

```sh
scripts/dev-setup.sh
.venv/bin/python -m pytest
```

The bridge tests do not require CrewAI itself. `pyproject.toml` adds `src` to
pytest's import path so the suite can run without an editable install.
