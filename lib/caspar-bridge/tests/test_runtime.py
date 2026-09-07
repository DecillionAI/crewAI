"""The runtime is the project's work record, and it is bounded."""

import asyncio

from decillion_caspar_bridge.runtime import CrewRuntime


def _runtime(sent):
    async def send(action, payload):
        sent.append((action, payload))
        return {"ok": True}

    return CrewRuntime("space-1", send)


def test_a_prompt_with_no_agents_answers_instead_of_failing_silently():
    sent = []
    runtime = _runtime(sent)
    asyncio.run(
        runtime.handle_prompt(
            {"runId": "r1", "prompt": "do the thing", "agents": []}
        )
    )
    actions = [payload for action, payload in sent if action == "crew/message"]
    assert len(actions) == 1
    assert actions[0]["kind"] == "answer"
    assert actions[0]["spaceId"] == "space-1"


def test_an_empty_prompt_is_dropped_rather_than_run():
    sent = []
    runtime = _runtime(sent)
    asyncio.run(runtime.handle_prompt({"runId": "r1", "prompt": "   "}))
    assert sent == []


def test_work_filters_by_agent_and_run():
    runtime = _runtime([])
    runtime._history.append({"runId": "r1", "agentProgramId": "a1"})
    runtime._history.append({"runId": "r2", "agentProgramId": "a2"})
    assert [r["runId"] for r in runtime.work(agent_program_id="a2")] == ["r2"]
    assert [r["runId"] for r in runtime.work(run_id="r1")] == ["r1"]
    # Newest first, so a member reading the history sees the latest turn.
    assert [r["runId"] for r in runtime.work()] == ["r2", "r1"]


def test_work_history_is_bounded():
    runtime = _runtime([])
    for i in range(500):
        runtime._history.append({"runId": f"r{i}", "agentProgramId": "a1"})
    # A sandbox that ran for a month must not hold every turn it ever ran.
    assert len(runtime.work()) == runtime._history.maxlen


def test_work_limit_is_applied():
    runtime = _runtime([])
    for i in range(10):
        runtime._history.append({"runId": f"r{i}", "agentProgramId": "a1"})
    assert len(runtime.work(limit=3)) == 3
