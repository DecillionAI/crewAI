"""Whether the process's three objects actually fit together.

This is the test that was missing. Everything else in this suite exercises one
object against fakes, and all of it passed while the process could not survive
its own startup: `__main__` asked the outbox for a method it does not have and
passed the state directory where the project id goes. The sandbox's restart loop
turned that into a project stuck at "Starting the tool server" forever, with the
agents on the node unable to reach their workspace and nothing anywhere saying
why.

So these assert the WIRING: that the graph builds, that a tool result is made
durable, and that liveness is not.
"""

from __future__ import annotations

import asyncio

from decillion_tool_server.__main__ import wire
from decillion_tool_server.config import BridgeConfig


def _config(**overrides) -> BridgeConfig:
    base = dict(
        gateway_url="ws://node:8076",
        space_id="space-1",
        topic="space:space-1",
        token="t",
        crew_home="/opt/crewai",
        log_level="INFO",
        state_dir="/data/.decillion/state",
        runtime_ref="abc",
    )
    base.update(overrides)
    return BridgeConfig(**base)


def _sender():
    sent: list[tuple[str, dict]] = []

    async def send(action: str, payload: dict) -> dict:
        sent.append((action, payload))
        return {"ok": True}

    return sent, send


def test_the_process_graph_builds():
    sent, send = _sender()
    outbox, server = wire(_config(), send)
    # Constructing it is the whole assertion: this raised AttributeError on the
    # real objects, in production, on every start.
    assert outbox.pending == 0
    assert server is not None
    assert sent == []


def test_a_tool_result_is_queued_durably_rather_than_sent_and_forgotten():
    sent, send = _sender()
    outbox, server = wire(_config(), send)

    asyncio.run(server._report({"callId": "c1", "ok": True, "result": "42"}))

    # On the project's volume, not on the wire: the socket may be down and the
    # sandbox may be replaced before it comes back, and the run is parked on a
    # frame paying for the wait.
    assert outbox.pending == 1
    assert sent == [], "a result must not depend on the socket being up at the moment it is produced"


def test_liveness_goes_straight_out_and_is_never_replayed():
    sent, send = _sender()
    outbox, server = wire(_config(), send)

    asyncio.run(server.heartbeat())

    # A heartbeat replayed from ten minutes ago would tell the node something
    # false, so it is not queued — it is sent, and allowed to fail.
    assert outbox.pending == 0
    assert [action for action, _ in sent] == ["crew/status"]


def test_an_announcement_travels_on_the_live_connection():
    sent, send = _sender()
    outbox, server = wire(_config(), send)

    asyncio.run(server.announce())

    # An announcement goes to `crew/bridge` exactly as a result does, so the two
    # are told apart by `fn`. It is what the node replays a project's held tool
    # calls against, and it means nothing on a connection that has since closed
    # — queueing it would hand the node a stale catalogue at an arbitrary later
    # moment.
    assert outbox.pending == 0
    assert [(action, payload.get("fn")) for action, payload in sent] == [("crew/bridge", "announce")]
