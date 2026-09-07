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
    assert [a["kind"] for a in actions] == ["answer", "usage"]
    assert actions[0]["spaceId"] == "space-1"
    # Nothing ran, so the meter is told zero rather than left to time the
    # payer's authorization out.
    assert actions[1]["promptTokens"] == 0


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


def test_a_failed_run_still_reports_what_it_used(monkeypatch):
    """A turn that raised still burned tokens, so it still has to be settled."""
    sent = []
    runtime = _runtime(sent)

    async def boom(*_args, **_kwargs):
        raise RuntimeError("the model refused")

    runtime._kickoff = boom
    # crewai is not installed in this environment, so the roster is stubbed:
    # what is under test is what happens AFTER a run fails, not how an agent is
    # constructed (that is test_roster's job).
    import decillion_caspar_bridge.runtime as runtime_module

    monkeypatch.setattr(runtime_module, "build_roster", lambda *_a, **_k: {"a1": object()})
    asyncio.run(
        runtime.handle_prompt(
            {
                "runId": "r1",
                "prompt": "do the thing",
                "agents": [{"programId": "a1", "name": "A", "role": "r"}],
                "agentProgramId": "a1",
            }
        )
    )
    kinds = [payload["kind"] for _action, payload in sent]
    assert "usage" in kinds
    usage = next(p for _a, p in sent if p["kind"] == "usage")
    assert usage["runId"] == "r1"
    assert usage["success"] is False
    assert usage["runtimeMs"] >= 0
    # The usage report is last: nothing settles a run before its answer is out.
    assert kinds[-1] == "usage"


def test_token_usage_is_read_from_whichever_shape_crewai_offers():
    from decillion_caspar_bridge.runtime import _token_usage

    class Output:
        token_usage = {"prompt_tokens": 11, "completion_tokens": 7}

    assert _token_usage(Output(), None) == {"promptTokens": 11, "completionTokens": 7}

    class Crew:
        class usage_metrics:  # noqa: N801 - mirrors CrewAI's attribute
            promptTokens = 3
            completionTokens = 4

    assert _token_usage(object(), Crew()) == {"promptTokens": 3, "completionTokens": 4}
    # An unknown shape reports zero rather than raising: a renamed counter must
    # never stop a run from being recorded.
    assert _token_usage(object(), object()) == {"promptTokens": 0, "completionTokens": 0}
