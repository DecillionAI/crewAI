"""The bridge process: one per project, running inside its Modal sandbox.

Started as the sandbox's entrypoint by `spaces/create`, under a restart loop —
so the sandbox lives exactly as long as its runtime, and a crash reconnects
rather than taking the project's machine down.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys

from .client import CasparBridgeClient
from .config import load_config
from .llm_proxy import LlmProxyServer
from .outbox import Outbox
from .runtime import CrewRuntime
from .tools import warm_catalog

logger = logging.getLogger("decillion_caspar_bridge")


async def _run() -> int:
    config = load_config()
    logging.basicConfig(
        level=getattr(logging, config.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if not config.configured:
        # Exiting non-zero rather than idling: the sandbox's restart loop will
        # try again, and the log says exactly what is missing.
        logger.error("bridge is not configured (%s)", config.describe())
        return 2
    logger.info("starting bridge: %s", config.describe())

    client: CasparBridgeClient | None = None

    async def send(action: str, payload: dict) -> dict:
        assert client is not None
        return await client.signal(action, payload)

    # The platform's model proxy. It runs before the runtime because an agent's
    # LLM is built against its base URL: the sandbox holds no provider key, and
    # every model call goes back out through the bridge to the `llm` creature,
    # which makes the real call and counts what it cost.
    async def call_creature(action: str, payload: dict) -> dict:
        assert client is not None
        return await client.call_creature(action, payload)

    async def await_result(correlation_id: str, timeout: float) -> dict:
        assert client is not None
        return await client.await_creature_result(correlation_id, timeout)

    llm_proxy = LlmProxyServer(call_creature, port=config.llm_proxy_port)
    await llm_proxy.start()

    # Everything a run produces goes out through here: written to the project's
    # own volume first, delivered after, retried until the node accepts it. It is
    # started before the runtime so anything a previous process left undelivered
    # — the settlement report of a turn this sandbox was killed in the middle of,
    # most of all — is on its way before the first new prompt arrives.
    outbox = Outbox(config.space_id, send, call_creature)
    outbox.start()

    runtime = CrewRuntime(
        config.space_id, send, llm_proxy, call_creature, await_result, outbox
    )

    async def on_update(key: str, data: dict) -> None:
        """Everything the project's creatures push to this bridge."""
        if key == "crew/prompt":
            await runtime.handle_prompt(data)
            return
        if key == "crew/work":
            # A member asked what the agents have been doing. The runtime is
            # the record, so it answers directly, correlated back to the
            # request that asked.
            await send(
                "crew/message",
                {
                    "kind": "reply",
                    "correlationId": str(data.get("correlationId") or ""),
                    "result": {
                        "runs": runtime.work(
                            str(data.get("agentProgramId") or ""),
                            str(data.get("runId") or ""),
                            int(data.get("limit") or 0),
                        ),
                        "status": runtime.status(),
                    },
                },
            )
            return
        if key == "creature/result":
            # A creature answering a call this sandbox made. The gateway only
            # acknowledged delivery, so this is where the answer actually
            # arrives; the correlation id says which caller was waiting.
            correlation_id = str(data.get("correlationId") or "")
            if not correlation_id or not client.resolve_creature_call(
                correlation_id, data.get("result")
            ):
                logger.debug("no caller waiting for %s", correlation_id or "(no id)")
            return
        if key == "crew/ping":
            # The creature is probing whether this project's runtime is
            # connected. Reaching this handler at all is the answer.
            await send("crew/status", {"ok": True, **runtime.status()})
            return
        logger.debug("ignoring update %s", key)

    client = CasparBridgeClient(config, on_update)

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:  # pragma: no cover - not all platforms
            pass

    serve = asyncio.create_task(client.run())

    async def announce() -> None:
        """Tell the project its runtime is up, once connected.

        This is also what asks for the work: the platform holds every prompt in
        a durable inbox, and a bridge announcing itself is what makes it replay
        whatever nobody has acknowledged. So a turn sent while this sandbox was
        asleep arrives moments after it wakes, rather than being a turn somebody
        was told to send again.
        """
        if await client.wait_connected(timeout=120):
            try:
                result = await send("crew/status", {"ok": True, **runtime.status()})
                replayed = (result or {}).get("replay") if isinstance(result, dict) else None
                logger.info(
                    "announced readiness for %s%s",
                    config.space_id,
                    f" (replaying {replayed})" if replayed else "",
                )
            except Exception:  # noqa: BLE001 - readiness is reported, not required
                logger.exception("could not announce readiness")

    async def warm_tools() -> None:
        """Build the tool catalogue before anyone prompts.

        Constructing it accepts the catalogue's own install prompts, so the
        first build genuinely installs packages and takes minutes. That cost is
        paid here, on a worker thread while the bridge is otherwise idle, rather
        than inside whichever prompt happened to arrive first.
        """
        try:
            count = await asyncio.to_thread(warm_catalog)
            logger.info("tool catalogue ready: %d tools", count)
        except Exception:  # noqa: BLE001 - the catalogue is a bonus, not the runtime
            logger.exception("could not build the tool catalogue")

    asyncio.create_task(announce())
    asyncio.create_task(warm_tools())

    await stop.wait()
    logger.info("shutting down")
    # One last chance to hand over what is already produced. Whatever does not
    # go now stays on disk and is replayed by the next process, so the bound is
    # a courtesy rather than a deadline anything depends on.
    await outbox.drain(timeout=15)
    await outbox.close()
    await llm_proxy.close()
    await client.close()
    serve.cancel()
    return 0


def main() -> int:
    try:
        return asyncio.run(_run())
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
