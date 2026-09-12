"""The tool-server process: one per project, running inside its Modal sandbox.

Started as the sandbox's entrypoint by `spaces/create`, under a restart loop —
so the sandbox lives exactly as long as this process, and a crash reconnects
rather than taking the project's machine down with it.

This used to start an agent runtime. It starts a tool runner: the agents live on
Caspar now, and what remains here is the catalogue and the workspace, which
genuinely cannot.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal
import sys

from typing import Awaitable, Callable

from .client import CasparBridgeClient
from .config import BridgeConfig, load_config
from .outbox import Outbox
from .state import stage
from .server import ToolServer

logger = logging.getLogger("decillion_tool_server")

#: How often to tell the node this machine is still here. Comfortably inside the
#: window the node treats as "this server went away", because a heartbeat is
#: much cheaper than the wake that missing one causes.
_HEARTBEAT_SECS = 60.0

#: The one thing this process says that must outlive it: a tool's RESULT. It is
#: the answer to work a run is parked on, and the run is paying for the wait.
#:
#: Told apart by `fn`, not by the creature it goes to: an announcement travels
#: to `crew/bridge` as well, and an announcement is only true while the socket
#: it was sent on is up.
_DURABLE = ("crew/bridge", "result")

#: What the process uses to reach the node: one action and its payload, answered.
Sender = Callable[[str, dict], Awaitable[dict]]


def wire(config: BridgeConfig, send: Sender) -> tuple[Outbox, ToolServer]:
    """Build the process's object graph.

    Separated from `_run` so it can be built without a socket, a loop or a
    sandbox — because the one thing no test covered was whether these three
    objects fit together, and they did not. `__main__` called an `Outbox`
    method that does not exist and passed the state directory where the project
    id goes; the process died on its first line of real work, every time, and
    the sandbox's restart loop hid it as "still starting". Forty passing tests
    said nothing about it.

    What a tool PRODUCED goes through the outbox: written to the project's own
    volume first, delivered after, retried until the node accepts it. A tool
    result dropped on a flaky socket is a run that waits out its frame lease for
    no reason — and the sandbox that computed it may be gone by then. The outbox
    finds its own directory (`DECILLION_STATE_DIR`, via `state_dir`), so it is
    given the project it belongs to and nothing else.
    """
    outbox = Outbox(config.space_id, send)

    async def report(action: str, payload: dict) -> dict:
        """Send one thing the tool server has to say.

        Two kinds, and they want opposite things. A tool RESULT is work already
        done: it must survive this process, so it is queued durably and retried
        until the node takes it. Liveness — announcing, heartbeats — is only
        true at the instant it is sent, and an announcement is what the node
        replays a project's backlog against, so it must travel on the live
        connection rather than through a queue; replaying either later would
        tell the node something false.
        """
        if (action, str(payload.get("fn") or "")) == _DURABLE:
            return {"ok": True, "eventId": outbox.post("toolresult", payload, action=action)}
        return await send(action, payload)

    server = ToolServer(
        report,
        config.space_id,
        runtime_ref=config.runtime_ref,
        reported_call_ids=outbox.pending_tool_call_ids,
    )
    return outbox, server


async def _run() -> int:
    config = load_config()
    logging.basicConfig(
        level=getattr(logging, config.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if not config.configured:
        # Exiting non-zero rather than idling: the sandbox's restart loop will
        # try again, and the log says exactly what is missing.
        logger.error("tool server is not configured (%s)", config.describe())
        stage(f"The tool server has no configuration ({config.describe()})")
        return 2
    logger.info("starting tool server: %s", config.describe())

    client: CasparBridgeClient | None = None

    async def send(action: str, payload: dict) -> dict:
        assert client is not None
        return await client.signal(action, payload)

    outbox, server = wire(config, send)

    async def on_update(key: str, payload: dict) -> None:
        """One packet pushed onto this project's topic."""
        if key == "tool/invoke":
            await server.on_request(payload)
            return
        # An unknown key is not an error. The node may publish things a given
        # version of this process does not handle yet, and a server that fell
        # over on one would be a server that cannot be upgraded independently.
        logger.debug("ignoring update on key %s", key)

    client = CasparBridgeClient(config, on_update)

    async def on_connected() -> None:
        """Announce readiness every time the socket comes back.

        Not just at startup: a subscription belongs to a connection and does not
        survive one closing, so a reconnect leaves the node believing this
        machine is gone.

        Nothing is done about the backlog here on purpose — the outbox's own
        worker retries with backoff for as long as it takes, so a reconnect
        needs no nudge, and blocking this path on a queue that empties only when
        the node is healthy would be the wrong thing to wait for.
        """
        await server.announce()
        # The bootstrap's last line is "Starting the tool server", so without
        # this a machine that is up reads exactly like one that never started.
        stage("This project's tools are connected and ready")

    client.on_connected = on_connected

    async def heartbeat() -> None:
        while True:
            await asyncio.sleep(_HEARTBEAT_SECS)
            with contextlib.suppress(Exception):
                await server.heartbeat()

    stopping = asyncio.Event()

    def _stop(*_args: object) -> None:
        stopping.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, _stop)

    async def warm_catalogue() -> None:
        """Build the catalogue, then tell the node it grew.

        OFF the startup path, and that is the whole point. Building says YES to
        the package installs several catalogue constructors ask for, so a cold
        machine spends minutes here — and while it did, this process had not
        connected, had not announced, and the project sat at "Starting the tool
        server" with a sandbox that was in fact running perfectly. Liveness must
        never queue behind an unbounded, network-dependent build.
        """
        warmed = await asyncio.to_thread(server.warm)
        logger.info("catalogue ready: %d tools", warmed)
        stage(f"Tool catalogue ready ({warmed} tools)")
        # The node keeps the last catalogue it was told about, so announcing
        # again simply replaces the workspace-only list with the full one.
        with contextlib.suppress(Exception):
            await server.announce(replay=False)

    # Begin delivering, which also replays whatever a previous process left on
    # the volume undelivered.
    outbox.start()

    tasks = [
        asyncio.create_task(client.run()),
        asyncio.create_task(heartbeat()),
        asyncio.create_task(warm_catalogue()),
    ]
    #await stopping.wait()

    #logger.info("shutting down")
    #for task in tasks:
    #    task.cancel()
    #with contextlib.suppress(asyncio.CancelledError):
    #    await asyncio.gather(*tasks, return_exceptions=True)
    #await outbox.close()
    #await client.close()
    return 0


def main() -> None:
    try:
        sys.exit(asyncio.run(_run()))
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
