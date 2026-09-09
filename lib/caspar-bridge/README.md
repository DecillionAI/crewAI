# decillion-caspar-bridge

A Decillion project's agents, running as a CrewAI crew inside that project's
Modal sandbox.

One bridge process runs per project. It is the sandbox's entrypoint (started by
the platform's `spaces/create`), and it is the only thing that knows about both
sides:

```
Caspar creature  ── publishUpdate ──►  bridge  ──►  CrewAI crew
CrewAI crew      ──►  bridge  ── /gateway/signal ──►  Caspar creature
```

## How it authenticates

The bridge holds no Caspar key. It presents a **bearer token** the project's
`spaces/create` minted and wrote into this sandbox, scoped to one topic
(`space:<id>`). The node checks it, so a bridge can only ever speak for its own
project, and the creature it reaches is the one the grant nominates — not one
the bridge names.

## Configuration

Read from `/etc/decillion/bridge.env`, written by the same call that created
the sandbox:

| Variable | Meaning |
|----------|---------|
| `CASPAR_GATEWAY_URL` | WebSocket URL of the Caspar node |
| `DECILLION_SPACE_ID` | The project this bridge serves |
| `DECILLION_BRIDGE_TOPIC` | Gateway topic (defaults to `space:<id>`) |
| `DECILLION_BRIDGE_TOKEN` | The bearer token — never logged, never sent back |
| `CREWAI_HOME` | Where the CrewAI checkout lives |
| `DECILLION_LOG_LEVEL` | Log level (default `INFO`) |
| `DECILLION_STATE_DIR` | Durable state (default `/data/.decillion`, the project's own volume) |

## What it does with a prompt

`crew/prompt` arrives with the project's roster attached, so a turn always runs
against the team as it stands at that moment. Each Decillion agent becomes a
CrewAI agent — role, goal and backstory come from the market listing's
descriptor, with the platform's universal instruction placed **before** the
agent's own persona — and the crew runs.

While it runs, CrewAI's event bus streams task and tool events back as
`kind=step` / `kind=toolcall` records tagged with the run. When it finishes, the
runtime posts exactly **one** `kind=answer`. That is the platform's one-writer
rule: the runtime runs the turn, so the runtime posts its answer, and nothing
writes the same row twice.

## Nothing a run produced is only in memory

A sandbox is not permanent: it sleeps after five idle minutes and the runtime
terminates it, and the bridge process restarts on a crash. So everything a run
produces is written to the project's own volume (`/data/.decillion`) before it
is sent, and it stays there until the node has accepted it.

* **The outbox** (`outbox.py`) holds steps, tool calls, answers, settlement
  reports and terminal events. Delivery is retried with backoff, unbounded, and
  anything a dead process left behind is replayed by the next one. Ordering is
  preserved, so an answer never lands ahead of the steps that produced it.
* **The settlement report is confirmed, not assumed.** `usage` goes out as a
  request/response call and is retried until `crew/message` says the run was
  actually billed. The gateway's delivery acknowledgement is not that answer,
  and reading it as though it were is how a rejected settlement stayed invisible.
* **Prompts are acknowledged.** The platform holds every prompt in a durable
  inbox and replays whatever no runtime claimed, so this bridge sends a
  `run-ack` the moment it takes a turn — and announcing readiness on connect is
  what asks for anything queued while the sandbox was asleep.
* **An answered question is acknowledged too** (`question-ack`), which is what
  retires it on the platform. Until then it stays answerable, so an answer given
  while this sandbox was gone is delivered when it comes back.

## Work history

CrewAI runs here, so the work history is here — `crew/work` is answered from
the runtime's own record of what each run did. That record is written to
`/data/.decillion/work`, so it survives the machine sleeping; the durable
authority is still the platform's run ledger and the project's signal log, and
this is the detailed cache in front of them.

## Running the tests

```
scripts/dev-setup.sh          # venv + test deps, then runs the suite
.venv/bin/python -m pytest    # afterwards
```

The package declares `crewai`, which pins Python to `<3.14`, so `pip install -e .`
refuses on a newer interpreter. The tests do not need it — every `crewai` import
in this package is lazy — so `pyproject.toml` puts `src` on pytest's
`pythonpath` instead of requiring an install, and naming a config file there
also stops collection from reaching for the crewAI workspace's own conftest.
