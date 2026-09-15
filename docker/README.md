# Deployment images

CrewAI is deployed to a Caspar node as **two images**, because the framework has
two jobs that must not run in the same place.

| Image | What is in it | Where it runs |
|-------|---------------|---------------|
| `crewai-engine` | `crewai`, `crewai-core`, `crewai-cli` | the Caspar node, as a docker creature |
| `crewai-sandbox` | the above **plus** `crewai-tools` and `tool-server` | each project's own Modal sandbox |

The split is the security boundary. The engine plans, delegates and calls
models; it never touches a project's files. The tools do nothing but touch a
project's files, its shell and its network — so they live on that project's own
machine, behind the tool server, reachable only through a bearer token the
platform minted for one topic and a fixed route list.

A tool call crosses between them:

```
engine ── signal ──► platform ── publishUpdate(space:<id>) ──► sandbox tool server
engine ◄── signal ── platform ◄── /gateway/signal ───────────┘
```

## Building

```sh
docker build -f docker/engine.Dockerfile  -t crewai-engine:dev  .
docker build -f docker/sandbox.Dockerfile -t crewai-sandbox:dev --build-arg RUNTIME_REF="$(git rev-parse HEAD)" .
```

Both build from the **checkout**, not from the index: the workspace packages pin
each other by exact version, so installing the local directories together is
what makes an image contain the code in this commit.

## Publishing

`.github/workflows/build-caspar-images.yml` builds and pushes both to
`ghcr.io/<owner>/crewai-engine` and `ghcr.io/<owner>/crewai-sandbox` on every
push that touches the packages or these Dockerfiles. Each gets a branch tag, an
immutable `sha-<commit>` tag, and `latest` on the default branch.

**A push is the release.** Autobot's deployer takes the engine tag as the base
image of its engine creature, and hands the sandbox tag to every space as the
image its machine boots — so neither has a version to bump by hand. Pin a
`sha-` tag instead when a deployment's machines should change only on request.

## `RUNTIME_REF`

The sandbox image records the commit it was built from at
`/opt/decillion/runtime-ref`. A machine decides whether to reinstall by
comparing the commit a fetch **resolved to** against what its volume records —
never the ref's name. A marker reading `main` would match itself forever, and
the machine would keep running the first commit it ever saw.

## The tool server needs no changes to serve another platform

`lib/tool-server` names its outbound route `crew/bridge` and listens for
`tool/invoke` on its topic. Neither is an address: a bridge grant carries a
`routes` map of `{action → programId}`, so the platform that mints the grant
decides which of *its* creatures each action reaches. Autobot points both
`crew/bridge` and `crew/status` at its own backend creature and the tool server
runs unmodified.

What the two sides exchange:

| Direction | Shape |
|-----------|-------|
| platform → sandbox | `publishUpdate` key `tool/invoke`, `{callId, tool, args}` |
| sandbox → platform | `{fn:"announce", spaceId, tools[], ref, replay}` |
| sandbox → platform | `{fn:"result", callId, ok, result\|error, durationMs}` |
| sandbox → platform | `{fn:"heartbeat", spaceId, at}` |

A result is sent through the durable outbox and retried until the node accepts
it, so the platform **must** answer a result signal on its correlation id.
Anything else is read as "not delivered", and the same result arrives again.
