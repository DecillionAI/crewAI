"""The outbox is what makes a run event survive this process.

Every claim these tests make is one the platform depends on: a step that is
persisted before it is sent, an answer that never overtakes the steps that led
to it, a settlement report that is retried until the node says it was actually
billed, and everything still on disk being replayed by the next process.
"""

from __future__ import annotations

import asyncio

import pytest

from decillion_tool_server.outbox import Outbox


def _collector():
    sent: list[dict] = []

    async def send(action, payload):
        sent.append(payload)
        return {"ok": True}

    return sent, send


def test_an_event_is_on_disk_before_anything_tries_to_send_it():
    _, send = _collector()
    outbox = Outbox("space-1", send)
    outbox.post("step", {"runId": "r1", "text": "thinking"})
    # Nothing has been delivered — the worker is not even started — and the
    # event already exists. That is the ordering the whole design turns on.
    assert outbox.pending == 1


def test_delivery_removes_the_record():
    async def main():
        sent, send = _collector()
        outbox = Outbox("space-1", send)
        outbox.post("answer", {"runId": "r1", "text": "done"})
        outbox.start()
        assert await outbox.drain(timeout=5)
        await outbox.close()
        return sent, outbox

    sent, outbox = asyncio.run(main())
    assert [event["kind"] for event in sent] == ["answer"]
    assert outbox.pending == 0


def test_a_failing_connection_retries_instead_of_dropping_the_event():
    async def main():
        attempts = {"n": 0}
        sent = []

        async def send(action, payload):
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise RuntimeError("connection closed")
            sent.append(payload)
            return {"ok": True}

        outbox = Outbox("space-1", send)
        outbox.post("usage", {"runId": "r1", "runtimeMs": 1000})
        outbox.start()
        assert await outbox.drain(timeout=10)
        await outbox.close()
        return attempts["n"], sent

    attempts, sent = asyncio.run(main())
    assert attempts == 3
    assert len(sent) == 1


def test_order_is_preserved_across_a_failure():
    async def main():
        sent = []
        failed = {"done": False}

        async def send(action, payload):
            if not failed["done"]:
                failed["done"] = True
                raise RuntimeError("connection closed")
            sent.append(payload)
            return {"ok": True}

        outbox = Outbox("space-1", send)
        outbox.post("step", {"runId": "r1", "text": "one"})
        outbox.post("step", {"runId": "r1", "text": "two"})
        outbox.post("answer", {"runId": "r1", "text": "final"})
        outbox.start()
        assert await outbox.drain(timeout=10)
        await outbox.close()
        return sent

    sent = asyncio.run(main())
    # The answer must never overtake the steps that produced it, even when the
    # first delivery fails and is retried.
    assert [event["text"] for event in sent] == ["one", "two", "final"]


def test_a_refused_settlement_is_retried_rather_than_believed():
    async def main():
        calls = {"n": 0}

        async def send(action, payload):
            return {"ok": True}

        async def call(action, payload):
            calls["n"] += 1
            if calls["n"] == 1:
                # What a rejected settlement looks like. The gateway still
                # ACCEPTED the packet; only the creature's own answer says the
                # run was not billed.
                return {"ok": False, "error": "could not reserve the pool slice"}
            return {"ok": True, "settled": True}

        outbox = Outbox("space-1", send, call)
        outbox.post("usage", {"runId": "r1", "runtimeMs": 1000})
        outbox.start()
        assert await outbox.drain(timeout=10)
        await outbox.close()
        return calls["n"]

    assert asyncio.run(main()) == 2


def test_usage_goes_through_the_confirming_channel_not_the_posting_one():
    async def main():
        posted, called = [], []

        async def send(action, payload):
            posted.append(payload["kind"])
            return {"ok": True}

        async def call(action, payload):
            called.append(payload["kind"])
            return {"ok": True}

        outbox = Outbox("space-1", send, call)
        outbox.post("step", {"runId": "r1"})
        outbox.post("usage", {"runId": "r1"})
        outbox.start()
        assert await outbox.drain(timeout=10)
        await outbox.close()
        return posted, called

    posted, called = asyncio.run(main())
    assert posted == ["step"]
    assert called == ["usage"]


def test_what_a_dead_process_left_behind_is_replayed_by_the_next_one(tmp_path):
    async def main():
        directory = tmp_path / "outbox"
        directory.mkdir()

        async def refuse(action, payload):
            raise RuntimeError("the sandbox went away")

        dying = Outbox("space-1", refuse, directory=directory)
        dying.post("usage", {"runId": "r1", "runtimeMs": 4000})
        dying.post("run-terminal", {"runId": "r1", "status": "succeeded"})
        # The process ends here without ever delivering either event.

        sent, send = _collector()
        reborn = Outbox("space-1", send, directory=directory)
        reborn.start()
        assert await reborn.drain(timeout=10)
        await reborn.close()
        return sent

    sent = asyncio.run(main())
    # Both events, in the order the dead process produced them. The settlement
    # report is the one that matters: without this it was work already delivered
    # that nobody was ever billed for.
    assert [event["kind"] for event in sent] == ["usage", "run-terminal"]


def test_every_event_carries_an_id_the_creature_can_dedupe_on():
    _, send = _collector()
    outbox = Outbox("space-1", send)
    first = outbox.post("step", {"runId": "r1"})
    second = outbox.post("step", {"runId": "r1"})
    assert first and second and first != second


def test_pending_tool_results_expose_the_calls_they_already_cover():
    _, send = _collector()
    outbox = Outbox("space-1", send)
    outbox.post(
        "toolresult",
        {"fn": "result", "callId": "c1", "ok": True, "result": "done"},
        action="crew/bridge",
    )
    outbox.post("step", {"runId": "r1"})

    assert outbox.pending_tool_call_ids == {"c1"}
