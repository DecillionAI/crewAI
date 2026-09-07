"""The bridge's connection to the Caspar node.

One connection does both directions:

* **inbound** — the node pushes `update` frames on the topics this bridge
  subscribed to. That is how a creature reaches a program that has no Caspar
  identity of its own.
* **outbound** — `/gateway/signal` calls, which the node delivers to the
  creature its grant nominates (`crew/message`).

Reconnection is not optional here: a sandbox lives for days, the node restarts,
and a bridge that gave up would leave a project's agents unreachable with no
symptom except silence. So the loop reconnects with backoff and re-subscribes
every time, because a subscription belongs to a connection and does not survive
one closing.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any, Awaitable, Callable

import websockets

from .config import BridgeConfig
from .protocol import ACK_FRAME, decode_frame, encode_request

logger = logging.getLogger(__name__)

UpdateHandler = Callable[[str, dict], Awaitable[None]]

#: Reconnect backoff, in seconds. Capped so a long outage does not turn into an
#: hours-long silence once the node comes back.
_BACKOFF_START = 1.0
_BACKOFF_CAP = 30.0

#: How long to wait for a response to one action before giving up on it. The
#: node answers gateway actions from state, so this is generous, not tight.
_REQUEST_TIMEOUT = 30.0


class CasparBridgeClient:
    """A reconnecting client for the gateway subscription channel."""

    def __init__(self, config: BridgeConfig, on_update: UpdateHandler) -> None:
        self._config = config
        self._on_update = on_update
        self._socket: websockets.ClientConnection | None = None
        self._pending: dict[str, asyncio.Future] = {}
        self._connected = asyncio.Event()
        self._closing = False

    # ── lifecycle ────────────────────────────────────────────────────────

    async def run(self) -> None:
        """Connect, subscribe and serve until closed, reconnecting as needed."""
        backoff = _BACKOFF_START
        while not self._closing:
            try:
                async with websockets.connect(
                    self._config.gateway_url,
                    max_size=None,
                    ping_interval=20,
                    ping_timeout=20,
                ) as socket:
                    self._socket = socket
                    logger.info("connected to %s", self._config.gateway_url)
                    backoff = _BACKOFF_START
                    await self._subscribe()
                    self._connected.set()
                    await self._read_loop(socket)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - any failure is a retry
                logger.warning("caspar connection lost: %s", exc)
            finally:
                self._connected.clear()
                self._socket = None
                self._fail_pending("connection closed")
            if self._closing:
                break
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, _BACKOFF_CAP)

    async def close(self) -> None:
        self._closing = True
        socket = self._socket
        if socket is not None:
            await socket.close()

    async def wait_connected(self, timeout: float | None = None) -> bool:
        try:
            await asyncio.wait_for(self._connected.wait(), timeout)
            return True
        except asyncio.TimeoutError:
            return False

    # ── actions ──────────────────────────────────────────────────────────

    async def _subscribe(self) -> None:
        """Bind this connection to the project's topic.

        A subscription belongs to a connection, so this runs on every connect,
        not once at startup.
        """
        result = await self.call(
            "/gateway/subscribe",
            {"token": self._config.token, "topics": [self._config.topic]},
        )
        topics = result.get("topics") or []
        if not topics:
            raise RuntimeError(f"subscribe was refused: {result}")
        logger.info("subscribed to %s", ", ".join(topics))

    async def signal(
        self,
        action: str,
        payload: dict[str, Any],
        correlation_id: str = "",
    ) -> dict:
        """Call one of the project's creature actions.

        The target creature is decided by the grant, not by this call — a
        bridge cannot address anything except the handler its project's token
        nominates.
        """
        return await self.call(
            "/gateway/signal",
            {
                "token": self._config.token,
                "topic": self._config.topic,
                "action": action,
                "correlationId": correlation_id,
                "payload": payload,
            },
        )

    async def call(self, path: str, payload: dict[str, Any]) -> dict:
        socket = self._socket
        if socket is None:
            raise RuntimeError("not connected to the caspar node")
        packet_id = uuid.uuid4().hex
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[packet_id] = future
        try:
            await socket.send(
                encode_request(path, packet_id, json.dumps(payload).encode("utf-8"))
            )
            return await asyncio.wait_for(future, _REQUEST_TIMEOUT)
        finally:
            self._pending.pop(packet_id, None)

    # ── inbound ──────────────────────────────────────────────────────────

    async def _read_loop(self, socket: websockets.ClientConnection) -> None:
        async for message in socket:
            if isinstance(message, str):
                message = message.encode("utf-8")
            try:
                frame = decode_frame(message)
            except ValueError as exc:
                logger.warning("dropping malformed frame: %s", exc)
                continue

            if frame["kind"] == "response":
                # The node holds the next response until this arrives.
                await socket.send(ACK_FRAME)
                self._resolve(frame)
                continue

            key = frame["key"]
            try:
                data = json.loads(frame["payload"].decode("utf-8") or "null")
            except ValueError:
                logger.warning("update %s carried a non-JSON payload", key)
                continue
            if not isinstance(data, dict):
                data = {"value": data}
            # Handled on its own task so one slow turn cannot stall the socket
            # (and with it every other project message).
            asyncio.create_task(self._dispatch(key, data))

    async def _dispatch(self, key: str, data: dict) -> None:
        try:
            await self._on_update(key, data)
        except Exception:  # noqa: BLE001 - a handler failure must not kill the loop
            logger.exception("handler for %s failed", key)

    def _resolve(self, frame: dict) -> None:
        future = self._pending.get(frame["packetId"])
        if future is None or future.done():
            return
        try:
            body = json.loads(frame["payload"].decode("utf-8") or "{}")
        except ValueError:
            body = {}
        if frame["code"] != 0:
            future.set_exception(
                RuntimeError(f"caspar refused the call: {body or frame['code']}")
            )
            return
        future.set_result(body if isinstance(body, dict) else {"result": body})

    def _fail_pending(self, reason: str) -> None:
        for future in list(self._pending.values()):
            if not future.done():
                future.set_exception(RuntimeError(reason))
        self._pending.clear()
