"""An OpenAI-compatible endpoint that holds no key.

CrewAI (through LiteLLM) talks to a model over HTTP, so the cheapest way to
take the platform's credentials out of this sandbox is to give it an HTTP
endpoint that looks exactly like OpenAI's and forwards every call over the
bridge's existing socket to the `llm` creature. The creature holds the key,
makes the real request, and records what the provider said it cost.

Two things follow from that, and both are the point:

* **No credential is in this container.** An agent that gets a shell here finds
  a loopback URL and a placeholder key, not the platform's OpenAI account.
* **Token counts are not self-reported.** The thing being metered is this
  process; a number it reports about itself is not evidence. The creature reads
  the count off the provider's own response.

The server binds loopback only. Nothing outside the sandbox can reach it, and
nothing inside it needs to be trusted to — it has nothing worth taking.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

#: The action the creature side answers on. Overridden by whatever the crew
#: creature names in `llmProxy.action`, so moving the creature does not need a
#: new sandbox image.
DEFAULT_ACTION = "llm/chat"

#: Loopback only, always. This endpoint is an internal detail of one sandbox.
_HOST = "127.0.0.1"


class LlmProxyServer:
    """Serves `POST /v1/chat/completions` by asking the platform to make the call."""

    def __init__(
        self,
        call: Callable[[str, dict], Awaitable[dict]],
        *,
        port: int = 8788,
        action: str = DEFAULT_ACTION,
    ) -> None:
        #: A REQUEST/RESPONSE call to a creature — not a send. The gateway
        #: acknowledges delivery and returns; the completion comes back
        #: separately, and this waits for it (see `CasparBridgeClient.
        #: call_creature`).
        self._call = call
        self._port = port
        self._action = action
        self._server: asyncio.AbstractServer | None = None
        #: Decillion provider id per model, filled from the roster: LiteLLM
        #: sends only a model string, and the creature needs to know which
        #: vendor's key to spend.
        self._providers: dict[str, str] = {}

    @property
    def base_url(self) -> str:
        return f"http://{_HOST}:{self._port}/v1"

    def set_action(self, action: str) -> None:
        if action:
            self._action = action

    def bind_model(self, model: str, provider: str) -> None:
        """Remember which provider a model belongs to."""
        if model and provider:
            self._providers[model] = provider

    async def start(self) -> None:
        if self._server is not None:
            return
        self._server = await asyncio.start_server(self._handle, _HOST, self._port)
        logger.info("llm proxy listening on %s", self.base_url)

    async def close(self) -> None:
        if self._server is None:
            return
        self._server.close()
        try:
            await self._server.wait_closed()
        finally:
            self._server = None

    # ── the request ──────────────────────────────────────────────────────

    async def _handle(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            request_line = await reader.readline()
            if not request_line:
                return
            parts = request_line.decode("latin-1").split()
            path = parts[1] if len(parts) > 1 else ""
            length = 0
            while True:
                line = await reader.readline()
                if not line or line in (b"\r\n", b"\n"):
                    break
                name, _, value = line.decode("latin-1").partition(":")
                if name.strip().lower() == "content-length":
                    try:
                        length = int(value.strip())
                    except ValueError:
                        length = 0
            raw = await reader.readexactly(length) if length > 0 else b""

            if not path.endswith("/chat/completions"):
                await self._write(writer, 404, {"error": {"message": "not found"}})
                return
            try:
                body = json.loads(raw.decode("utf-8") or "{}")
            except (ValueError, UnicodeDecodeError):
                await self._write(writer, 400, {"error": {"message": "invalid json body"}})
                return

            status, payload = await self._forward(body)
            await self._write(writer, status, payload)
        except (asyncio.IncompleteReadError, ConnectionError):
            return
        except Exception:  # noqa: BLE001 - one bad request must not stop the server
            logger.exception("llm proxy request failed")
            try:
                await self._write(writer, 500, {"error": {"message": "proxy failure"}})
            except Exception:  # noqa: BLE001
                pass
        finally:
            try:
                writer.close()
            except Exception:  # noqa: BLE001
                pass

    async def _forward(self, body: dict[str, Any]) -> tuple[int, dict]:
        model = str(body.get("model") or "")
        # LiteLLM prefixes a model with its provider ("openai/gpt-4o"); the
        # creature wants the vendor's own id, and the Decillion provider comes
        # from the roster rather than from that prefix — an OpenAI-compatible
        # gateway is spelled "openai" by LiteLLM and is a different vendor.
        bare = model.split("/", 1)[1] if "/" in model else model
        provider = self._providers.get(model) or self._providers.get(bare) or ""
        if not provider:
            return 400, {
                "error": {"message": f"no Decillion provider is bound to model {model!r}"}
            }
        request = {
            "provider": provider,
            "model": bare,
            "messages": body.get("messages") or [],
        }
        for key in ("temperature", "top_p", "stop", "tools", "tool_choice", "response_format"):
            if body.get(key) is not None:
                request[key] = body[key]
        if body.get("max_tokens"):
            request["maxTokens"] = body["max_tokens"]
        # Streaming is asked for on the CREATURE's side, not answered on this
        # one: a creature replies to a signal once, so there is no stream to
        # hand back. It matters anyway, because several providers report token
        # usage only on a stream's final event, and an unmetered call is one
        # the platform bought and cannot bill.
        if body.get("stream"):
            request["stream"] = True

        reply = await self._call(self._action, request)
        result = reply.get("result") if isinstance(reply.get("result"), dict) else reply
        if not isinstance(result, dict):
            return 502, {"error": {"message": "the platform returned no response"}}
        if result.get("ok") is False:
            status = int(result.get("status") or 502)
            message = str(result.get("error") or "the model call was refused")
            return (status if 400 <= status < 600 else 502), {"error": {"message": message}}
        response = result.get("response")
        if not isinstance(response, dict):
            return 502, {"error": {"message": "the platform returned no completion"}}
        return 200, response

    async def _write(self, writer: asyncio.StreamWriter, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        head = (
            f"HTTP/1.1 {status} OK\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\n"
            "Connection: close\r\n\r\n"
        ).encode("latin-1")
        writer.write(head + body)
        await writer.drain()
