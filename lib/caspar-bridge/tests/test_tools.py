"""What an agent may reach, and how a tool's declaration is read.

These cover the parts that decide *what a tool can touch* and *what shape its
arguments are*, both of which are wrong-answer-shaped rather than
exception-shaped: a path that escapes the project's folder still writes a file,
and a mis-read argument list still builds a tool — it just builds one the model
cannot call correctly. Neither shows up as a crash, so both are tested here.

CrewAI itself is not imported: building an Agent needs the whole framework, and
nothing below depends on it.
"""

from __future__ import annotations

import os
import sys

import pytest

from decillion_caspar_bridge import tools


def test_a_path_stays_inside_the_project(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "WORKSPACE_ROOT", str(tmp_path))
    assert tools._resolve("notes/plan.md") == (tmp_path / "notes/plan.md").resolve()
    assert tools._resolve(".") == tmp_path.resolve()
    # A leading slash is read as "from the project root", not as the machine's.
    assert tools._resolve("/notes/plan.md") == (tmp_path / "notes/plan.md").resolve()


@pytest.mark.parametrize("escape", ["../outside.txt", "notes/../../outside.txt", "../../etc/passwd"])
def test_a_path_that_climbs_out_is_refused(tmp_path, monkeypatch, escape):
    monkeypatch.setattr(tools, "WORKSPACE_ROOT", str(tmp_path))
    with pytest.raises(ValueError):
        tools._resolve(escape)


def test_output_is_clipped_with_the_size_it_dropped():
    monkeypatched = "x" * (tools.MAX_OUTPUT_CHARS + 25)
    clipped = tools._clip(monkeypatched)
    assert len(clipped) < len(monkeypatched)
    assert "25 more characters" in clipped
    assert tools._clip("short") == "short"


def test_declared_args_reads_a_tools_own_list_shape():
    command = {
        "name": "write",
        "args": [
            {"name": "repo", "description": "owner/name"},
            {"name": "path", "description": "path in the repository"},
        ],
    }
    assert tools._declared_args(command) == [
        ("repo", "owner/name"),
        ("path", "path in the repository"),
    ]


def test_declared_args_reads_the_registrys_map_shape():
    """The synthetic `help` entry declares a map, not a list."""
    command = {"name": "help", "args": {"command": {"type": "STRING", "desc": "a command name"}}}
    assert tools._declared_args(command) == [("command", "a command name")]


def test_a_command_with_no_arguments_declares_none():
    assert tools._declared_args({"name": "status"}) == []
    assert tools._declared_args({"name": "status", "args": None}) == []


def test_tool_names_are_callable_identifiers():
    assert tools._tool_slug("GitHub", "setShared") == "github_setshared"
    assert tools._tool_slug("Zapier ✨", "run") == "zapier_run"
    # Never empty: a model cannot call a tool with no name.
    assert tools._tool_slug("", "") == "tool"


def test_a_creatures_reply_drops_the_envelope():
    rendered = tools._render(
        {"ok": True, "namespace": "github", "action": "invoke", "function": "status", "connected": True}
    )
    assert "connected" in rendered
    assert "namespace" not in rendered
    # A reply that was ALL envelope still says something.
    assert tools._render({"ok": True, "namespace": "github"}) == "Done."


def test_build_tools_survives_a_sandbox_without_crewai():
    """A runtime that cannot build tools still runs the turn.

    crewai is absent here, so every builder raises on import. The contract is
    that `build_tools` reports an empty set rather than propagating — an agent
    with no tools is degraded, an agent whose turn raised is broken.
    """
    assert tools.build_tools([{"action": "tool/1@global", "commands": []}], None, None) == []


def test_the_catalogue_can_be_turned_off(monkeypatch):
    monkeypatch.setattr(tools, "_CATALOG_ENABLED", False)
    monkeypatch.setattr(tools, "_CATALOG_CACHE", None)
    assert tools.catalog_tools() == []


def test_a_command_a_person_must_run_is_not_offered_to_agents():
    """`agents: false` is how a tool keeps a command to people.

    Only an explicit false excludes: a registry entry written before the flag
    existed says nothing, and must keep every command it declared rather than
    silently losing them all.
    """
    assert tools._agent_may_call({"name": "read"}) is True
    assert tools._agent_may_call({"name": "connect", "agents": False}) is False
    assert tools._agent_may_call({"name": "read", "agents": True}) is True


def test_a_tool_missing_its_required_key_is_not_offered():
    """The framework says which key a tool needs; a tool without it is dropped.

    Read from the instance rather than from a list here, so a catalogue that
    grows a tool does not need this file to grow with it.
    """

    class Var:
        def __init__(self, name, required):
            self.name = name
            self.required = required

    class Tool:
        env_vars = [Var("DEFINITELY_NOT_SET_KEY", True)]

    class Optional:
        env_vars = [Var("DEFINITELY_NOT_SET_KEY", False)]

    class Dicts:
        env_vars = [{"name": "DEFINITELY_NOT_SET_KEY", "required": True}]

    assert tools._missing_env(Tool()) == ["DEFINITELY_NOT_SET_KEY"]
    assert tools._missing_env(Optional()) == []
    assert tools._missing_env(Dicts()) == ["DEFINITELY_NOT_SET_KEY"]
    assert tools._missing_env(object()) == []


def test_a_constructor_that_asks_to_install_is_told_yes():
    """Catalogue constructors prompt on stdin; there is no terminal here.

    The answer is yes, every time — declining a constructor means giving up its
    tool, and a sandbox is a disposable machine built to run this project's
    agents. A blocking read would instead hang a project's first turn with
    nothing in the log to explain it.
    """
    with tools._auto_approve():
        print("this must not reach the real stdout")
        assert input("install it? [y/N] ") == "y"
        assert input("and this one? [y/N] ") == "y"


def test_approvals_run_out_rather_than_spinning_forever():
    """A constructor that never accepts the answer must not loop for ever.

    Past the supply the reads hit EOF, which surfaces as a failure to build —
    handled, and the tool is simply not offered.
    """
    with tools._auto_approve():
        for _ in range(tools._MAX_APPROVALS):
            assert input("? ") == "y"
        with pytest.raises(EOFError):
            input("? ")


def test_a_tool_whose_schema_a_provider_refuses_is_not_offered():
    """One unusable schema costs the agent EVERY tool, so it is caught here.

    A model call carries all of an agent's tools in one request. OpenAI refuses
    the whole request over a single `allOf`, so the agent makes no tool calls at
    all — and then reports work it never did, which is the failure this filter
    exists to prevent.
    """

    class Bad:
        @staticmethod
        def model_json_schema():
            return {"properties": {"config": {"allOf": [{"$ref": "#/$defs/X"}]}}}

    class Good:
        @staticmethod
        def model_json_schema():
            return {"properties": {"path": {"type": "string"}}}

    class Undescribable:
        @staticmethod
        def model_json_schema():
            raise TypeError("no schema")

    assert tools._unsupported_schema(type("T", (), {"args_schema": Bad})()) == ["allOf"]
    assert tools._unsupported_schema(type("T", (), {"args_schema": Good})()) == []
    assert tools._unsupported_schema(type("T", (), {"args_schema": Undescribable})()) == [
        "an unreadable schema"
    ]
    # A tool with no schema at all takes no arguments; that is fine.
    assert tools._unsupported_schema(object()) == []


def test_the_question_tool_is_only_built_for_a_real_turn():
    """A question has to name its run and its author, so it needs the turn.

    Without one there is nothing to attribute the question to and nothing for
    the answer to come back to, so the tool is simply not offered.
    """
    calls = []

    async def call(action, payload):  # pragma: no cover - never reached here
        calls.append((action, payload))
        return {}

    # crewai is absent in this environment, so every builder fails and the list
    # is empty either way — what is asserted is that asking for a turn does not
    # raise, and that omitting one is equally safe.
    assert tools.build_tools([], call, None) == []
    assert tools.build_tools([], call, None, {"runId": "r-1"}) == []


def test_the_install_target_is_set_up_and_put_back(tmp_path, monkeypatch):
    """`uv add` needs a project to edit and a venv to install into.

    Without both, "yes" installs nothing — which is how the approval looked
    like it worked while every package still failed. The working directory is
    process-global, so what matters as much is that it is restored.
    """
    monkeypatch.setattr(tools, "_INSTALL_PROJECT", tmp_path / "scratch")
    monkeypatch.delenv("UV_PROJECT_ENVIRONMENT", raising=False)
    before = os.getcwd()

    with tools._installable():
        assert os.getcwd() == str((tmp_path / "scratch").resolve())
        assert (tmp_path / "scratch" / "pyproject.toml").exists()
        assert os.environ["UV_PROJECT_ENVIRONMENT"] == sys.prefix

    assert os.getcwd() == before
    assert "UV_PROJECT_ENVIRONMENT" not in os.environ
