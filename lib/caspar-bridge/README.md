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

## Work history

CrewAI runs here, so the work history is here — `crew/work` is answered from
the runtime's own bounded record of what each run did. There is no second copy
kept anywhere else to drift.
