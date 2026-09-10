"""The tool server: what it runs, what it refuses, and what it reports.

This process is what is LEFT of the old agent runtime, and most of these tests
exist to pin down the difference. It runs tools that need this machine and
nothing else: no crew, no roster, no model calls, and no provider credential
anywhere near it.
"""

from __future__ import annotations

import asyncio

import pytest

from decillion_tool_server import tools as tools_mod
from decillion_tool_server.server import ToolServer


class _FakeTool:
    """A tool shaped the way `crewai_tools` and the workspace tools are."""

    def __init__(self, name, description="a tool", result="ok", raises=None, schema=None):
        self.name = name
        self.description = description
        self._result = result
        self._raises = raises
        self.args_schema = schema
        self.calls = []

    def run(self, **kwargs):
        self.calls.append(kwargs)
        if self._raises is not None:
            raise self._raises
        return self._result


@pytest.fixture
def catalogue(monkeypatch):
    """Install a known tool set, so these tests do not depend on what pip did."""
    installed = {}

    def _set(workspace=(), catalog=()):
        installed["workspace"] = list(workspace)
        installed["catalog"] = list(catalog)
        monkeypatch.setattr(tools_mod, "workspace_tools", lambda: installed["workspace"])
        monkeypatch.setattr(tools_mod, "catalog_tools", lambda: installed["catalog"])
        return installed

    return _set


# ── Running a tool ──────────────────────────────────────────────────────────


def test_a_tool_runs_and_returns_what_it_said(catalogue):
    tool = _FakeTool("FileReadTool", result="# README")
    catalogue(workspace=[tool])
    assert tools_mod.run_tool("FileReadTool", {"path": "README.md"}) == "# README"
    assert tool.calls == [{"path": "README.md"}]


def test_a_missing_tool_is_an_answer_not_an_exception(catalogue):
    # The model chose a name. Being told which names are real is something it
    # can act on; a stack trace ends the turn instead.
    catalogue(workspace=[_FakeTool("FileReadTool")], catalog=[_FakeTool("SerperDevTool")])
    out = tools_mod.run_tool("NoSuchTool", {})
    assert out.startswith("Error:")
    assert "FileReadTool" in out and "SerperDevTool" in out


def test_wrong_arguments_tell_the_model_what_the_tool_wanted(catalogue):
    class _Schema:
        @staticmethod
        def model_json_schema():
            return {"type": "object", "properties": {"path": {"type": "string"}}}

    catalogue(workspace=[_FakeTool("FileReadTool", raises=TypeError("unexpected keyword 'file'"), schema=_Schema)])
    out = tools_mod.run_tool("FileReadTool", {"file": "x"})
    assert "wrong arguments" in out
    assert "path" in out, "a fixable mistake must say how to fix it"


def test_a_projects_own_tool_wins_a_name_collision(catalogue):
    # A catalogue tool called FileReadTool must never shadow the one that reads
    # THIS project's files — the agent asked about this project.
    mine = _FakeTool("FileReadTool", result="mine")
    theirs = _FakeTool("FileReadTool", result="theirs")
    catalogue(workspace=[mine], catalog=[theirs])
    assert tools_mod.run_tool("FileReadTool", {}) == "mine"


def test_a_huge_result_is_clipped_rather_than_sent_whole(catalogue):
    catalogue(workspace=[_FakeTool("Big", result="x" * 500_000)])
    out = tools_mod.run_tool("Big", {})
    assert len(out) < 500_000, "a signal frame has a size limit"


# ── The manifest ────────────────────────────────────────────────────────────


def test_the_manifest_describes_every_tool_this_machine_has(catalogue):
    class _Schema:
        @staticmethod
        def model_json_schema():
            return {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}

    catalogue(workspace=[_FakeTool("FileReadTool")], catalog=[_FakeTool("Search", schema=_Schema)])
    manifest = {entry["name"]: entry for entry in tools_mod.tool_manifest()}

    assert set(manifest) == {"FileReadTool", "Search"}
    assert manifest["Search"]["parameters"]["required"] == ["q"]
    # The node needs to know these run on the machine, not on the node.
    assert all(entry["kind"] == "sandbox" for entry in manifest.values())


def test_a_tool_with_an_unreadable_schema_still_appears(catalogue):
    class _Broken:
        @staticmethod
        def model_json_schema():
            raise RuntimeError("nope")

    catalogue(catalog=[_FakeTool("Odd", schema=_Broken)])
    manifest = tools_mod.tool_manifest()
    assert manifest[0]["name"] == "Odd", "a tool that runs must be offered even if it describes itself badly"
    assert manifest[0]["parameters"] == {"type": "object", "properties": {}}


# ── The server ──────────────────────────────────────────────────────────────


class _Recorder:
    def __init__(self):
        self.sent = []

    async def __call__(self, action, payload):
        self.sent.append((action, payload))
        return {"ok": True}


@pytest.mark.asyncio
async def test_announcing_reports_the_catalogue(catalogue):
    catalogue(workspace=[_FakeTool("FileReadTool")])
    send = _Recorder()
    server = ToolServer(send, "s1", runtime_ref="abc123")

    await server.announce()

    action, payload = send.sent[0]
    assert action == "crew/bridge"
    assert payload["fn"] == "announce"
    assert payload["spaceId"] == "s1"
    assert payload["ref"] == "abc123"
    # The catalogue travels WITH the announcement: asking for it separately
    # would be a round trip on the critical path of every first prompt.
    assert [t["name"] for t in payload["tools"]] == ["FileReadTool"]


@pytest.mark.asyncio
async def test_a_request_runs_and_reports_against_its_call_id(catalogue):
    catalogue(workspace=[_FakeTool("FileReadTool", result="contents")])
    send = _Recorder()
    server = ToolServer(send, "s1")

    await server.on_request({"callId": "c1", "tool": "FileReadTool", "args": {"path": "x"}})
    await asyncio.sleep(0)  # let the task run
    for _ in range(20):
        if send.sent:
            break
        await asyncio.sleep(0.01)

    action, payload = send.sent[0]
    assert action == "crew/bridge"
    assert payload["fn"] == "result"
    assert payload["callId"] == "c1"
    assert payload["ok"] is True
    assert payload["result"] == "contents"
    assert payload["durationMs"] >= 0


@pytest.mark.asyncio
async def test_a_failing_tool_is_reported_as_a_failure_not_a_crash(catalogue):
    catalogue(workspace=[_FakeTool("Flaky", raises=RuntimeError("vendor is down"))])
    send = _Recorder()
    server = ToolServer(send, "s1")

    await server.on_request({"callId": "c1", "tool": "Flaky", "args": {}})
    for _ in range(20):
        if send.sent:
            break
        await asyncio.sleep(0.01)

    _, payload = send.sent[0]
    assert payload["ok"] is False
    assert "vendor is down" in payload["error"]


@pytest.mark.asyncio
async def test_a_request_with_no_call_id_is_ignored_rather_than_answered(catalogue):
    catalogue(workspace=[_FakeTool("T")])
    send = _Recorder()
    server = ToolServer(send, "s1")
    await server.on_request({"tool": "T", "args": {}})
    await asyncio.sleep(0.02)
    assert send.sent == [], "there is nowhere to send an answer that names no call"


@pytest.mark.asyncio
async def test_a_redelivered_request_does_not_run_the_tool_twice(catalogue):
    # At-least-once delivery: the node republishes anything it has not had an
    # answer for, including things that are simply still running.
    started = asyncio.Event()
    release = asyncio.Event()

    class _Slow(_FakeTool):
        def run(self, **kwargs):
            self.calls.append(kwargs)
            started.set()
            asyncio.run(asyncio.sleep(0))  # yield without blocking the loop thread
            release.wait() if hasattr(release, "wait") else None
            return "done"

    tool = _FakeTool("Slow", result="done")
    catalogue(workspace=[tool])
    send = _Recorder()
    server = ToolServer(send, "s1")

    await server.on_request({"callId": "c1", "tool": "Slow", "args": {}})
    await server.on_request({"callId": "c1", "tool": "Slow", "args": {}})
    for _ in range(20):
        if send.sent:
            break
        await asyncio.sleep(0.01)

    assert len(tool.calls) == 1, "a duplicate delivery must not run the tool again"
