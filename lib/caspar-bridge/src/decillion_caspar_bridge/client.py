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
import contextlib
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

#: How long to wait for a CREATURE's answer (see `call_creature`). This is a
#: model call, not a state read: it is bounded by the vendor, not by the node.
_CREATURE_CALL_TIMEOUT = 300.0


class CasparBridgeClient:
    """A reconnecting client for the gateway subscription channel."""

    def __init__(self, config: BridgeConfig, on_update: UpdateHandler) -> None:
        self._config = config
        self._on_update = on_update
        self._socket: websockets.ClientConnection | None = None
        self._pending: dict[str, asyncio.Future] = {}
        #: Callers waiting on a CREATURE's answer, keyed by the correlation id
        #: their call carried. Distinct from `_pending`, which tracks the
        #: node's own transport-level responses: the gateway acknowledges
        #: delivery immediately, and the creature's answer arrives later as an
        #: update on this project's topic.
        self._creature_calls: dict[str, asyncio.Future] = {}
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
                    # The reader must be running BEFORE the first request.
                    # `_subscribe` waits for a response, and the only thing that
                    # delivers one is `_read_loop` — so awaiting subscribe first
                    # waits for a reply nobody is listening for. It timed out
                    # after `_REQUEST_TIMEOUT`, every time, and the bridge sat in
                    # a connect / 30s / reconnect loop that never subscribed to
                    # anything. A project's agents were simply unreachable.
                    reader = asyncio.create_task(self._read_loop(socket))
                    try:
                        await self._subscribe()
                        self._connected.set()
                        await reader
                    finally:
                        reader.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await reader
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - any failure is a retry
                # `str()` on an asyncio.TimeoutError is empty, which is how this
                # loop spent its life reporting "caspar connection lost: " and
                # nothing else. Name the type when the message is blank.
                detail = str(exc) or type(exc).__name__
                logger.warning("caspar connection lost: %s", detail)
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

    async def call_creature(
        self,
        action: str,
        payload: dict[str, Any],
        timeout: float = _CREATURE_CALL_TIMEOUT,
    ) -> dict:
        """Call a creature action and wait for its ANSWER.

        `signal` only tells you the node accepted the call: `/gateway/signal`
        delivers and returns, and the caller is not a Caspar identity that
        anything can signal back to. A creature that has something to say
        publishes it on this project's topic under `creature/result`, tagged
        with the correlation id generated here — which is what makes this a
        request/response call rather than a send.

        Without it, a caller reads the gateway's acknowledgement as though it
        were the creature's reply: the model proxy did exactly that and
        answered every completion with "the platform returned no completion".
        """
        correlation_id = uuid.uuid4().hex
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._creature_calls[correlation_id] = future
        try:
            ack = await self.signal(
                action, {**payload, "correlationId": correlation_id}, correlation_id
            )
            if isinstance(ack, dict) and ack.get("ok") is False:
                raise RuntimeError(f"{action} was refused: {ack.get('error') or ack}")
            return await asyncio.wait_for(future, timeout)
        finally:
            self._creature_calls.pop(correlation_id, None)

    async def await_creature_result(self, correlation_id: str, timeout: float) -> Any:
        """Wait for a result published under an id this client did not mint.

        `call_creature` covers the ordinary case: ask, and wait for the answer
        to that call. Some answers arrive under a DIFFERENT id, because they
        come from somewhere else entirely — a question is answered by a person,
        minutes later, and the creature says up front which id that answer will
        carry. Waiting on it is the same machinery, entered from the other end.
        """
        if not correlation_id:
            raise ValueError("a correlation id is required to wait for a result")
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._creature_calls[correlation_id] = future
        try:
            return await asyncio.wait_for(future, timeout)
        finally:
            self._creature_calls.pop(correlation_id, None)

    def resolve_creature_call(self, correlation_id: str, result: Any) -> bool:
        """Hand a creature's answer to the call waiting for it."""
        future = self._creature_calls.get(correlation_id)
        if future is None or future.done():
            return False
        future.set_result(result if isinstance(result, dict) else {"result": result})
        return True

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
        # Both tables: a creature's answer comes back over this connection too,
        # so a caller waiting on one must not wait out its whole timeout after
        # the socket has gone.
        for table in (self._pending, self._creature_calls):
            for future in list(table.values()):
                if not future.done():
                    future.set_exception(RuntimeError(reason))
            table.clear()
