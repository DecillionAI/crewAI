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
    # The acknowledgement comes FIRST, and before anything slow: it is what
    # claims the queued prompt AND what takes the payer's slice, so a run that
    # cannot be funded stops before it builds a tool or calls a model.
    assert [a["kind"] for a in actions] == ["run-ack", "answer", "usage", "run-terminal"]
    assert actions[0]["runId"] == "r1"
    assert actions[1]["spaceId"] == "space-1"
    # Nothing ran, so the meter is told zero rather than left to time the
    # payer's authorization out.
    assert actions[2]["promptTokens"] == 0


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
    # Settlement follows the answer and the explicit terminal event closes the
    # durable server ledger last.
    assert kinds[-2:] == ["usage", "run-terminal"]


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


# ── The two entry points ─────────────────────────────────────────────────────
#
# A project is reached in two ways and they must not collapse into one: `@lead`
# hands a whole objective to the CREW, and any other `@agent` addresses that
# agent alone. These exercise the decision without building a real crew — what
# is under test is which agents a turn is for and how they are arranged, not
# CrewAI itself.

class _FakeProcess:
    sequential = "sequential"
    hierarchical = "hierarchical"


class _FakeTask:
    def __init__(self, description, expected_output, agent=None):
        self.description = description
        self.expected_output = expected_output
        self.agent = agent


class _FakeCrew:
    def __init__(self, agents, tasks, process, verbose=False, manager_agent=None):
        self.agents = agents
        self.tasks = tasks
        self.process = process
        self.manager_agent = manager_agent


def _specs():
    return [
        {"programId": "lead-1", "username": "lead", "lead": True, "goal": "run the project"},
        {"programId": "eng-1", "username": "eng", "lead": False, "goal": "write code"},
        {"programId": "des-1", "username": "des", "lead": False, "goal": "design"},
    ]


class _FakeAgent:
    """Enough of a CrewAI agent to see which of them keep their tools."""

    def __init__(self, name, tools=("write_project_file",)):
        self.name = name
        self.tools = list(tools)

    backstory = ""

    def model_copy(self, update=None):
        clone = _FakeAgent(self.name, self.tools)
        clone.backstory = self.backstory
        for key, value in (update or {}).items():
            setattr(clone, key, value)
        return clone

    def __repr__(self):  # pragma: no cover - test output only
        return f"<{self.name} tools={self.tools}>"


def _roster():
    return {"lead-1": _FakeAgent("LEAD"), "eng-1": _FakeAgent("ENG"), "des-1": _FakeAgent("DES")}


def _build(selected, lead_id):
    by_id = {s["programId"]: s for s in _specs()}
    return CrewRuntime._build_crew(
        "ship the thing", _roster(), selected, by_id, lead_id,
        _FakeCrew, _FakeProcess, _FakeTask,
    )


def test_the_lead_runs_the_whole_crew_as_its_manager():
    crew = _build(["lead-1"], "lead-1")
    # One task for the objective, and the lead is the manager rather than one
    # more worker — that is what makes this collaboration and not a broadcast.
    assert crew.process == _FakeProcess.hierarchical
    assert crew.manager_agent.name == "LEAD"
    assert {a.name for a in crew.agents} == {"ENG", "DES"}
    assert len(crew.tasks) == 1
    # CrewAI refuses a manager that is also in `agents`, and a hierarchical task
    # must leave the executor to the manager.
    assert "LEAD" not in {a.name for a in crew.agents}
    assert crew.tasks[0].agent is None
    assert crew.tasks[0].description == "ship the thing"
    # And the manager holds NO tools. CrewAI refuses one that does, because a
    # manager with tools is a manager doing the work itself — while every
    # teammate it delegates to keeps them.
    assert crew.manager_agent.tools == []
    assert all(a.tools for a in crew.agents)


def test_mentioning_one_agent_still_addresses_only_that_agent():
    crew = _build(["eng-1"], "")
    assert crew.process == _FakeProcess.sequential
    assert [a.name for a in crew.agents] == ["ENG"]
    assert len(crew.tasks) == 1
    assert crew.tasks[0].agent.name == "ENG"
    assert crew.tasks[0].expected_output == "write code"
    # It is doing the work, so it has the tools to do it.
    assert crew.tasks[0].agent.tools


def test_a_lead_with_no_teammates_answers_by_itself():
    by_id = {s["programId"]: s for s in _specs()}
    crew = CrewRuntime._build_crew(
        "ship the thing", {"lead-1": _FakeAgent("LEAD")}, ["lead-1"], by_id, "lead-1",
        _FakeCrew, _FakeProcess, _FakeTask,
    )
    # A hierarchical crew with nobody to delegate to fails validation, and the
    # person asked for the work either way.
    assert crew.process == _FakeProcess.sequential
    assert [a.name for a in crew.agents] == ["LEAD"]
    assert crew.tasks[0].agent.name == "LEAD"
    # Nobody to delegate to means it does the work, so it keeps its tools —
    # the manager's empty toolset is a property of managing, not of being lead.
    assert crew.tasks[0].agent.tools


def test_the_lead_is_the_one_the_platform_marked():
    from decillion_caspar_bridge.runtime import _lead_of

    assert _lead_of(_specs(), _roster()) == "lead-1"
    # An agent that failed to build is not in the roster, and a lead that cannot
    # run must not silently swallow the turn.
    assert _lead_of(_specs(), {"eng-1": "ENG"}) == ""
    # No lead marked: every project that has none keeps working.
    assert _lead_of([{"programId": "eng-1", "lead": False}], {"eng-1": "ENG"}) == ""


def _selection(message, monkeypatch):
    """Which agents a turn is for, and whether it is a crew turn.

    `build_roster` and the crew run are replaced: what is under test is the
    decision `handle_prompt` makes, and building a real crew would require
    CrewAI and a model.
    """
    import decillion_caspar_bridge.runtime as rt

    monkeypatch.setattr(
        rt, "build_roster", lambda specs, *_a, **_k: {
            str(s["programId"]): f"AGENT:{s['programId']}" for s in specs
        }
    )
    seen = {}

    async def fake_kickoff(self, prompt, roster, selected, specs, run_id, thread_id, usage, lead_id=""):
        seen["selected"] = list(selected)
        seen["lead_id"] = lead_id
        return "done"

    monkeypatch.setattr(rt.CrewRuntime, "_kickoff", fake_kickoff)
    runtime = _runtime([])
    asyncio.run(runtime.handle_prompt({**message, "agents": _specs()}))
    return seen


def test_mentioning_the_lead_makes_it_a_crew_turn(monkeypatch):
    seen = _selection(
        {"runId": "r1", "prompt": "build the product", "mentions": [{"programId": "lead-1"}]},
        monkeypatch,
    )
    assert seen["lead_id"] == "lead-1"
    # The lead owns the record and writes the one answer.
    assert seen["selected"] == ["lead-1"]


def test_mentioning_a_teammate_is_that_agent_alone(monkeypatch):
    seen = _selection(
        {"runId": "r1", "prompt": "fix the bug", "mentions": [{"programId": "eng-1"}]},
        monkeypatch,
    )
    assert seen["lead_id"] == ""
    assert seen["selected"] == ["eng-1"]


def test_a_prompt_addressed_to_nobody_goes_to_the_lead(monkeypatch):
    seen = _selection({"runId": "r1", "prompt": "what should we do next?"}, monkeypatch)
    # A project-wide prompt is the lead's job when there is one — rather than
    # every agent answering the same question separately.
    assert seen["lead_id"] == "lead-1"
    assert seen["selected"] == ["lead-1"]


def test_a_project_with_no_lead_still_runs_its_whole_team(monkeypatch):
    import decillion_caspar_bridge.runtime as rt

    leaderless = [
        {"programId": "eng-1", "username": "eng", "lead": False},
        {"programId": "des-1", "username": "des", "lead": False},
    ]
    monkeypatch.setattr(
        rt, "build_roster", lambda specs, *_a, **_k: {
            str(s["programId"]): f"AGENT:{s['programId']}" for s in specs
        }
    )
    seen = {}

    async def fake_kickoff(self, prompt, roster, selected, specs, run_id, thread_id, usage, lead_id=""):
        seen["selected"] = list(selected)
        seen["lead_id"] = lead_id
        return "done"

    monkeypatch.setattr(rt.CrewRuntime, "_kickoff", fake_kickoff)
    asyncio.run(_runtime([]).handle_prompt({"runId": "r1", "prompt": "go", "agents": leaderless}))
    assert seen["lead_id"] == ""
    assert seen["selected"] == ["eng-1", "des-1"]


def test_the_lead_leads_even_when_mentioned_alongside_a_teammate(monkeypatch):
    seen = _selection(
        {
            "runId": "r1",
            "prompt": "ship it",
            "mentions": [{"programId": "eng-1"}, {"programId": "lead-1"}],
        },
        monkeypatch,
    )
    # Naming the lead is asking for the objective to be run, and the lead
    # decides who works on it — including whether that is the teammate also
    # named. Two overlapping runs for one message is the thing to avoid.
    assert seen["lead_id"] == "lead-1"
    assert seen["selected"] == ["lead-1"]


def test_a_step_names_the_teammate_that_did_it():
    from decillion_caspar_bridge.events import CrewEventForwarder

    class _Agent:
        role = "Designer"

    class _Task:
        description = "lay out the page"
        agent = _Agent()

    seen = []
    CrewEventForwarder(lambda kind, payload: seen.append((kind, payload)), "r1", "lead-1", "Lead")._step(
        "started", _Task()
    )
    kind, payload = seen[0]
    assert kind == "step"
    # The run belongs to the lead; the work does not.
    assert payload["agentProgramId"] == "lead-1"
    assert payload["agentName"] == "Lead"
    assert payload["actorName"] == "Designer"


def test_a_step_with_no_named_agent_claims_none():
    from decillion_caspar_bridge.events import CrewEventForwarder

    seen = []
    CrewEventForwarder(lambda kind, payload: seen.append((kind, payload)), "r1", "lead-1")._step(
        "started", None
    )
    assert seen[0][1]["agentName"] == ""


def test_a_step_uses_the_teammates_stable_program_identity():
    from decillion_caspar_bridge.events import CrewEventForwarder

    class _Agent:
        role = "Research role"

    class _Task:
        description = "verify sources"
        agent = _Agent()

    seen = []
    CrewEventForwarder(
        lambda kind, payload: seen.append((kind, payload)),
        "r1",
        "lead-1",
        "Lead",
        actors={id(_Task.agent): ("research-1", "Researcher")},
    )._step("started", _Task())
    payload = seen[0][1]
    assert payload["agentProgramId"] == "lead-1"
    assert payload["agentName"] == "Lead"
    assert payload["actorProgramId"] == "research-1"
    assert payload["actorName"] == "Researcher"


def test_forwarder_filters_other_runs_and_unregisters_every_handler():
    from decillion_caspar_bridge.events import CrewEventForwarder

    ours = object()
    theirs = object()
    forwarder = CrewEventForwarder(lambda *_a: None, "r1", "lead-1", sources=[ours])
    assert forwarder._owns(ours, object()) is True
    assert forwarder._owns(theirs, object()) is False

    removed = []

    class _Bus:
        def off(self, event_type, handler):
            removed.append((event_type, handler))

    handler = lambda *_a: None
    forwarder._bus = _Bus()
    forwarder._registered = [(str, handler), (int, handler)]
    forwarder.unregister()
    assert removed == [(str, handler), (int, handler)]
    assert forwarder._registered == []


def test_a_question_is_attributed_to_the_agent_that_was_addressed():
    from decillion_caspar_bridge.runtime import _asking_agent

    specs = [
        {"programId": "lead-1", "lead": True, "name": "Orbit Lead"},
        {"programId": "eng-1", "lead": False, "name": "Engineer"},
    ]
    # Addressed directly: that agent asks.
    assert _asking_agent({"agentProgramId": "eng-1"}, specs) == {
        "agentProgramId": "eng-1",
        "agentName": "Engineer",
    }
    # Addressed to the project: the lead speaks for it.
    assert _asking_agent({}, specs)["agentProgramId"] == "lead-1"
    # No lead: the first agent on the project.
    assert _asking_agent({}, [{"programId": "eng-1", "name": "Engineer"}])["agentProgramId"] == "eng-1"
    # Nobody at all: empty, which the creature refuses rather than posting a
    # message from no one.
    assert _asking_agent({}, [])["agentProgramId"] == ""


def test_handoff_mentions_only_resolve_unambiguous_prose_handles():
    from decillion_caspar_bridge.runtime import _handoff_targets

    specs = [
        {"programId": "research-1", "username": "researcher", "name": "Researcher"},
        {"programId": "build-1", "username": "builder", "name": "Builder"},
    ]
    text = """Please @researcher verify the market.
```css
@media (width > 10px) {}
```
Install `@types/node`, email me@example.com, and see https://x.test/@builder.
Thanks @builder!"""
    assert [s["programId"] for s in _handoff_targets(text, specs)] == ["research-1"]


def test_an_agents_mention_records_the_referral_and_starts_nobody():
    """A mention written by an AGENT is a reference, not a request.

    Agents collaborate inside the crew — the lead is CrewAI's manager and its
    own delegation picks who does each step, within the one turn the person
    asked for. So an answer naming a colleague is describing work that has
    already been shared, and launching that colleague again would do it twice.

    This used to mint a delegated quote and start a separately billed child
    run, which is why an answer with no billing pool attached posted "this run
    has no delegated billing pool for a server-side hand-off" instead of
    collaborating at all.
    """
    sent = []
    runtime = _runtime(sent)
    called = []

    async def _call(action, payload):
        called.append((action, payload))
        return {"ok": True}

    runtime._call = _call
    asyncio.run(
        runtime._note_mentions(
            [
                {"programId": "lead-1", "username": "lead", "name": "Lead"},
                {"programId": "research-1", "username": "researcher", "name": "Researcher"},
            ],
            ["lead-1"],
            {
                "runId": "parent-run", "jobId": "job-1", "threadId": "main",
                "agentProgramId": "lead-1", "agentName": "Lead",
            },
            "@researcher please verify these findings.",
        )
    )

    # Nothing was quoted and nothing was started.
    assert called == []
    # The referral is still visible in the run's trail, and it is not a failure.
    notice = sent[0][1]
    assert notice["kind"] == "step"
    assert notice["status"] == "completed"
    assert "@researcher" in notice["text"]


def test_an_answer_naming_nobody_says_nothing():
    sent = []
    runtime = _runtime(sent)
    asyncio.run(
        runtime._note_mentions(
            [{"programId": "lead-1", "username": "lead", "name": "Lead"}],
            ["lead-1"],
            {"runId": "parent-run", "agentProgramId": "lead-1", "agentName": "Lead"},
            "All done — no help needed.",
        )
    )
    assert sent == []


def test_the_work_record_survives_the_sandbox_being_replaced(tmp_path, monkeypatch):
    """A sleeping machine is a pause, not amnesia.

    `crew/work` used to be answered from an in-memory deque alone, so every
    sandbox — which sleeps after five idle minutes — came back having forgotten
    everything its agents had done.
    """
    monkeypatch.setenv("DECILLION_STATE_DIR", str(tmp_path / "state"))
    dying = _runtime([])
    dying._save_work_record(
        {"runId": "r1", "agentProgramId": "a1", "status": "done", "startedAt": 1.0}
    )
    dying._save_work_record(
        {"runId": "r2", "agentProgramId": "a2", "status": "done", "startedAt": 2.0}
    )

    # The sandbox is terminated and a new one comes up on the same volume.
    reborn = _runtime([])
    assert [row["runId"] for row in reborn.work()] == ["r2", "r1"]


def test_an_answered_question_is_acknowledged_so_the_platform_can_retire_it():
    """The platform holds an answered question until the runtime says it landed.

    Publishing an answer and removing the question at the same time is what lost
    both when the bridge had already gone.
    """
    sent = []
    runtime = _runtime(sent)
    runtime._loop = asyncio.new_event_loop()
    try:
        runtime._acknowledge_question("q-1")
        # Scheduled onto the loop the socket lives on; run it to completion.
        runtime._loop.run_until_complete(asyncio.sleep(0))
    finally:
        runtime._loop.close()
    # With no outbox this posts directly; with one it is durable. Either way the
    # acknowledgement carries the question it is for.
    posted = [payload for action, payload in sent if action == "crew/message"]
    assert any(p.get("kind") == "question-ack" and p.get("questionId") == "q-1" for p in posted)


def test_run_events_go_through_the_outbox_when_there_is_one(tmp_path, monkeypatch):
    """With an outbox, nothing a run produces is only in memory."""
    monkeypatch.setenv("DECILLION_STATE_DIR", str(tmp_path / "state"))
    from decillion_caspar_bridge.outbox import Outbox

    async def refuse(action, payload):
        raise RuntimeError("the node is not reachable")

    async def main():
        outbox = Outbox("space-1", refuse)
        runtime = CrewRuntime("space-1", refuse, outbox=outbox)
        await runtime.handle_prompt({"runId": "r1", "prompt": "do the thing", "agents": []})
        return outbox

    outbox = asyncio.run(main())
    # The node refused every delivery and not one event was lost: the
    # acknowledgement, the answer, the usage report and the terminal event are
    # all on disk for the next process to send.
    assert outbox.pending == 4


def test_a_run_the_platform_refuses_to_fund_never_starts():
    """A refusal at the acknowledgement stops the turn before it spends anything.

    The reservation is made by the creature this acknowledgement reaches, so
    `stop` is the platform saying the payer's pool cannot cover the run. Going
    ahead would deliver work nobody can be charged for.
    """
    sent = []

    async def send(action, payload):
        sent.append((action, payload))
        return {"ok": True}

    async def call(action, payload):
        sent.append((action, payload))
        return {"ok": False, "stop": True, "error": "this run could not be funded"}

    runtime = CrewRuntime("space-1", send, call=call)
    asyncio.run(
        runtime.handle_prompt({"runId": "r1", "prompt": "do the thing", "agents": []})
    )
    kinds = [payload.get("kind") for _, payload in sent]
    assert kinds == ["run-ack"]
    # No answer, no usage, no terminal event: the platform already closed the run.


def test_an_unreachable_platform_is_not_a_refusal():
    """A failed acknowledgement must not cancel work somebody paid for."""
    sent = []

    async def send(action, payload):
        sent.append((action, payload))
        return {"ok": True}

    async def call(action, payload):
        raise RuntimeError("the node is not reachable")

    runtime = CrewRuntime("space-1", send, call=call)
    asyncio.run(
        runtime.handle_prompt({"runId": "r1", "prompt": "do the thing", "agents": []})
    )
    kinds = [payload.get("kind") for _, payload in sent]
    # The acknowledgement falls back to the durable outbox path and the turn runs.
    assert kinds[0] == "run-ack"
    assert "answer" in kinds


def test_the_crew_delegating_is_posted_into_the_chat_as_a_mention():
    """When the lead hands work to a teammate, the project sees it happen.

    Agents collaborate inside a CrewAI hierarchical crew, and that conversation
    is the interesting part of a multi-agent turn. It is posted as a mention so
    it reads the way the people on the project talk — and it starts nothing,
    because the teammate is already doing the work.
    """
    from decillion_caspar_bridge.events import CrewEventForwarder

    class _Agent:
        role = "Lead role"

    class _Event:
        tool_name = "Delegate work to coworker"
        tool_args = {"coworker": "Research role", "task": "check the pricing page"}
        output = ""
        agent = _Agent()

    seen = []
    lead = _Agent()
    forwarder = CrewEventForwarder(
        lambda kind, payload: seen.append((kind, payload)),
        "r1",
        "lead-1",
        "Lead",
        actors={id(lead): ("lead-1", "Lead")},
        handles={"research role": "researcher"},
    )
    event = _Event()
    event.agent = lead
    forwarder._tool("started", event)

    kinds = [kind for kind, _ in seen]
    assert kinds == ["toolcall", "answer"]
    answer = seen[1][1]
    assert answer["interim"] is True
    # Addressed by the handle the project uses, not CrewAI's internal role.
    assert answer["text"] == "@researcher check the pricing page"
    assert answer["agentProgramId"] == "lead-1"

    # The reply comes back as the teammate's own work; posting the hand-off
    # again on the way out would read as it happening twice.
    seen.clear()
    forwarder._tool("finished", event)
    assert [kind for kind, _ in seen] == ["toolcall"]


def test_a_led_turn_reports_what_each_agent_spent():
    """One run, several agents — and every creator is owed for their own.

    A led turn is a single billed run under the lead, but the work inside it is
    done by whichever teammates the lead delegated to. Pricing the whole fee as
    the lead's minutes would pay the lead's creator for everybody's work.
    """
    from decillion_caspar_bridge.events import CrewEventForwarder

    class _Agent:
        def __init__(self, role):
            self.role = role

    class _Task:
        def __init__(self, agent):
            self.description = "do the thing"
            self.agent = agent

    designer, researcher = _Agent("Designer"), _Agent("Researcher")
    forwarder = CrewEventForwarder(
        lambda kind, payload: None,
        "r1",
        "lead-1",
        "Lead",
        actors={id(designer): ("design-1", "Designer"), id(researcher): ("research-1", "Researcher")},
    )

    one, two = _Task(designer), _Task(researcher)
    forwarder._step("started", one)
    forwarder._step("completed", one)
    forwarder._step("started", two)
    forwarder._step("failed", two)

    split = forwarder.agent_runtime_ms()
    assert set(split) == {"design-1", "research-1"}
    assert all(ms >= 0 for ms in split.values())

    # A task nobody is credited with is not billed to anybody.
    forwarder._step("started", _Task(_Agent("Unknown")))
    assert "" not in forwarder.agent_runtime_ms()
