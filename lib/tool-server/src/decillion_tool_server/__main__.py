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

from .client import CasparBridgeClient
from .config import load_config
from .outbox import Outbox
from .server import ToolServer

logger = logging.getLogger("decillion_tool_server")

#: How often to tell the node this machine is still here. Comfortably inside the
#: window the node treats as "this server went away", because a heartbeat is
#: much cheaper than the wake that missing one causes.
_HEARTBEAT_SECS = 60.0


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
        return 2
    logger.info("starting tool server: %s", config.describe())

    client: CasparBridgeClient | None = None

    async def send(action: str, payload: dict) -> dict:
        assert client is not None
        return await client.signal(action, payload)

    # Everything this process reports goes through the outbox: written to the
    # project's own volume first, delivered after, retried until the node
    # accepts it. A tool result dropped on a flaky socket is a run that waits out
    # its frame lease for no reason.
    outbox = Outbox(config.state_dir, send)
    server = ToolServer(outbox.send, config.space_id, runtime_ref=config.runtime_ref)

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
        machine is gone. Announcing again is also what replays the backlog.
        """
        await server.announce()
        await outbox.flush()

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

    # Build the catalogue BEFORE announcing. Saying yes to the tool installers
    # means the first build can take minutes, and announcing an empty catalogue
    # would offer an agent nothing and then quietly grow the list under it.
    warmed = await asyncio.to_thread(server.warm)
    logger.info("catalogue ready: %d tools", warmed)

    tasks = [asyncio.create_task(client.run()), asyncio.create_task(heartbeat())]
    await stopping.wait()

    logger.info("shutting down")
    for task in tasks:
        task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await asyncio.gather(*tasks, return_exceptions=True)
    await client.close()
    return 0


def main() -> None:
    try:
        sys.exit(asyncio.run(_run()))
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
