"""The tools a Decillion agent can actually use.

An agent with no tools can only talk. It says it saved the file, and nothing
was saved; it says it checked the repository, and it checked nothing. Every
tool here exists to close that gap, and they come from three places:

* **The project's machine.** The bridge runs *inside* the project's Modal
  sandbox, so reading and writing the project's files is a local operation, not
  a round trip. The root is `/data` — the sandbox's persistent volume, and the
  exact directory `spaces/files` serves to the file explorer — so what an agent
  writes is what a person sees in the Files panel, and what survives the
  machine sleeping.

* **The project's Caspar tools.** GitHub, Zapier, and anything registered
  later. These are creatures on the node: the sandbox holds no Caspar identity,
  so a call goes out over the bridge's socket to the tool's own dispatcher,
  which authorizes it from the project's bearer token (see the dispatcher's
  bridge prologue in `scripts/gen_endpoints.py`). Each of a tool's registered
  commands becomes one tool here, with the arguments the registry declares —
  the same rows the client turns into `@tool` autocomplete, so an agent and a
  person are offered exactly the same capabilities.

* **The CrewAI tool catalogue.** `crewai_tools` ships ~100 tools. Most need a
  vendor's API key or a constructor argument this platform has no value for, so
  the catalogue is filtered by whether a tool can be BUILT here: import it,
  construct it, and offer it only if both worked. Offering a tool that raises
  the moment an agent picks it is worse than not offering it — the agent burns
  a turn discovering what the platform already knew.

Nothing here decides what an agent *should* do. It decides what it *can*.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import logging
import os
import re
import subprocess
import sys
import threading
from concurrent.futures import TimeoutError as FuturesTimeout
from pathlib import Path
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

#: The project's own directory on its machine. This is the sandbox's persistent
#: volume and the root `spaces/files` reads, so it is the one place where an
#: agent's work is both visible to the project and durable across a sleep.
#: Anything written elsewhere in the container is lost when the machine stops.
WORKSPACE_ROOT = os.environ.get("DECILLION_WORKSPACE", "/data")

#: How long one shell command may run before it is killed. Long enough for an
#: install or a test run, short enough that a hung command does not hold the
#: turn open until the model's own timeout.
SHELL_TIMEOUT_SECONDS = 300

#: How much of a command's output (or a file's contents) to hand back. A tool
#: result goes into the next prompt, so an unbounded read is a bill as much as
#: a mistake.
MAX_OUTPUT_CHARS = 20_000

#: Set `DECILLION_CATALOG_TOOLS=off` to run with only the workspace and the
#: project's own tools — useful when a model's context is tight, since every
#: offered tool costs prompt tokens on every call.
_CATALOG_ENABLED = os.environ.get("DECILLION_CATALOG_TOOLS", "on").strip().lower() not in {
    "off",
    "0",
    "false",
    "no",
}


# ── the project's machine ────────────────────────────────────────────────────


def _resolve(path: str) -> Path:
    """One path inside the project's directory.

    Resolved against the workspace and refused if it climbs out. The sandbox is
    the security boundary — an agent that can run a shell can reach the whole
    container either way — so this is not what keeps the machine safe. It is
    what keeps the project's work in the project's folder, where the Files panel
    shows it and the volume keeps it, instead of scattered through a container
    that is thrown away on the next sleep.
    """
    root = Path(WORKSPACE_ROOT).resolve()
    candidate = (root / str(path or ".").lstrip("/")).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"{path} is outside the project's files")
    return candidate


def _clip(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    return text[:MAX_OUTPUT_CHARS] + f"\n… [{len(text) - MAX_OUTPUT_CHARS} more characters]"


def workspace_tools() -> list[Any]:
    """Read, write and run things on the project's own machine."""
    from crewai.tools import BaseTool
    from pydantic import BaseModel, Field

    class ReadArgs(BaseModel):
        path: str = Field(description="Path of the file, relative to the project folder")

    class WriteArgs(BaseModel):
        path: str = Field(description="Path of the file, relative to the project folder")
        content: str = Field(description="The complete contents to write")

    class AppendArgs(BaseModel):
        path: str = Field(description="Path of the file, relative to the project folder")
        content: str = Field(description="Text to add to the end of the file")

    class ListArgs(BaseModel):
        path: str = Field(default=".", description="Folder to list, relative to the project folder")

    class ShellArgs(BaseModel):
        command: str = Field(description="The shell command to run in the project folder")

    class ReadFile(BaseTool):
        name: str = "read_project_file"
        description: str = (
            "Read a file from the project's folder. Use this before editing a file, "
            "so you change what is actually there."
        )
        args_schema: type[BaseModel] = ReadArgs

        def _run(self, path: str) -> str:
            try:
                target = _resolve(path)
            except ValueError as exc:
                return f"Error: {exc}"
            if not target.is_file():
                return f"Error: {path} does not exist in the project's files"
            try:
                return _clip(target.read_text(encoding="utf-8", errors="replace"))
            except OSError as exc:
                return f"Error: could not read {path}: {exc}"

    class WriteFile(BaseTool):
        name: str = "write_project_file"
        description: str = (
            "Write a file into the project's folder, creating it and any missing "
            "folders. This is how you deliver work: a file written here is what "
            "the team sees in the project's Files panel. Overwrites the file."
        )
        args_schema: type[BaseModel] = WriteArgs

        def _run(self, path: str, content: str) -> str:
            try:
                target = _resolve(path)
            except ValueError as exc:
                return f"Error: {exc}"
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(str(content), encoding="utf-8")
            except OSError as exc:
                return f"Error: could not write {path}: {exc}"
            return f"Wrote {len(str(content))} characters to {path}"

    class AppendFile(BaseTool):
        name: str = "append_project_file"
        description: str = (
            "Add text to the end of a file in the project's folder, creating it if "
            "it does not exist. Use this for a log or a running document rather "
            "than rewriting the whole file."
        )
        args_schema: type[BaseModel] = AppendArgs

        def _run(self, path: str, content: str) -> str:
            try:
                target = _resolve(path)
            except ValueError as exc:
                return f"Error: {exc}"
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("a", encoding="utf-8") as handle:
                    handle.write(str(content))
            except OSError as exc:
                return f"Error: could not append to {path}: {exc}"
            return f"Added {len(str(content))} characters to {path}"

    class ListFiles(BaseTool):
        name: str = "list_project_files"
        description: str = (
            "List what is in a folder of the project. Use this first to find out "
            "what the project already contains, so you build on it rather than "
            "duplicating a teammate's work."
        )
        args_schema: type[BaseModel] = ListArgs

        def _run(self, path: str = ".") -> str:
            try:
                target = _resolve(path)
            except ValueError as exc:
                return f"Error: {exc}"
            if not target.is_dir():
                return f"Error: {path} is not a folder in the project's files"
            rows = sorted(
                f"{entry.name}/" if entry.is_dir() else f"{entry.name} ({entry.stat().st_size} bytes)"
                for entry in target.iterdir()
            )
            return _clip("\n".join(rows)) if rows else "(the folder is empty)"

    class RunShell(BaseTool):
        name: str = "run_shell_command"
        description: str = (
            "Run a shell command on the project's machine, in the project's folder. "
            "Use it to run code, tests, or any command-line tool. Returns the "
            "command's output; a command that takes longer than five minutes is "
            "stopped."
        )
        args_schema: type[BaseModel] = ShellArgs

        def _run(self, command: str) -> str:
            try:
                completed = subprocess.run(  # noqa: S602 - a shell is the point
                    str(command),
                    shell=True,
                    cwd=WORKSPACE_ROOT,
                    capture_output=True,
                    text=True,
                    timeout=SHELL_TIMEOUT_SECONDS,
                )
            except subprocess.TimeoutExpired:
                return f"Error: the command did not finish within {SHELL_TIMEOUT_SECONDS} seconds"
            except OSError as exc:
                return f"Error: could not run the command: {exc}"
            parts = []
            if completed.stdout:
                parts.append(completed.stdout)
            if completed.stderr:
                parts.append(f"[stderr]\n{completed.stderr}")
            if completed.returncode != 0:
                parts.append(f"[exit code {completed.returncode}]")
            return _clip("\n".join(parts)) if parts else "(the command produced no output)"

    return [ReadFile(), WriteFile(), AppendFile(), ListFiles(), RunShell()]


# ── the project's Caspar tools ───────────────────────────────────────────────


def _tool_slug(*parts: str) -> str:
    """A tool name a model can call: letters, digits and underscores only."""
    joined = "_".join(str(p or "").strip() for p in parts if str(p or "").strip())
    # Runs are collapsed, not replaced one-for-one: a name ending in punctuation
    # would otherwise meet the separator and produce a doubled underscore, and
    # two tools whose names differ only there would look like different tools to
    # a person reading the trail and the same one to nobody.
    slug = re.sub(r"_+", "_", re.sub(r"[^0-9a-zA-Z_]+", "_", joined)).strip("_").lower()
    return slug or "tool"


def _declared_args(command: dict[str, Any]) -> list[tuple[str, str]]:
    """One command's parameters, in either shape the registry stores them.

    A tool declares `args` as a list of `{name, description}`; the synthetic
    `help` command the registry adds declares a map of `name -> {desc}`. Both
    are read here rather than normalised on the node, because the node's copy
    is what the client also renders and changing it would change that too.
    """
    args = command.get("args")
    out: list[tuple[str, str]] = []
    if isinstance(args, list):
        for item in args:
            if isinstance(item, dict) and item.get("name"):
                out.append((str(item["name"]), str(item.get("description") or item.get("desc") or "")))
            elif isinstance(item, str) and item:
                out.append((item, ""))
    elif isinstance(args, dict):
        for name, spec in args.items():
            if not name:
                continue
            if isinstance(spec, dict):
                out.append((str(name), str(spec.get("desc") or spec.get("description") or "")))
            else:
                out.append((str(name), ""))
    return out


def _agent_may_call(command: dict[str, Any]) -> bool:
    """Whether a registered command is one an AGENT may run.

    The tool says so. `agents: false` marks the commands that manage the
    connection rather than use it — starting a sign-in, which answers with a URL
    only a person can open, or revoking a credential a person granted. Only an
    explicit false excludes: a command that says nothing is ordinary work, and a
    registry written before the flag existed must not lose everything it
    declared.
    """
    return command.get("agents") is not False


def creature_tools(
    specs: list[dict[str, Any]],
    call: Callable[[str, dict], Awaitable[dict]],
    loop: asyncio.AbstractEventLoop,
    timeout: float = 120.0,
) -> list[Any]:
    """The project's attached Caspar tools, one CrewAI tool per command.

    `call` is the bridge's request/response call to a creature and `loop` is the
    bridge's event loop: CrewAI runs a turn on a worker thread, so each call is
    handed to the loop and waited on from that thread. Doing it the other way —
    a fresh event loop per call — would open a second connection to the node for
    every tool an agent uses.
    """
    from pydantic import Field, create_model

    tools: list[Any] = []
    for spec in specs or []:
        action = str(spec.get("action") or "").strip()
        if not action:
            # A tool with no route is one this runtime cannot reach — a metered
            # container tool, which is executed by the node and not by us. It is
            # skipped rather than offered and failed.
            continue
        tool_name = str(spec.get("name") or "tool")
        for command in spec.get("commands") or []:
            if not isinstance(command, dict):
                continue
            fn = str(command.get("name") or "").strip()
            if not fn or fn == "help":
                # `help` describes the tool to a person typing in chat. An agent
                # has the same information in these descriptions already.
                continue
            if not _agent_may_call(command):
                continue
            declared = _declared_args(command)
            fields: dict[str, Any] = {
                name: (str, Field(default="", description=desc or f"{name} for {tool_name} {fn}"))
                for name, desc in declared
            }
            args_schema = create_model(f"{_tool_slug(tool_name, fn)}_args", **fields)  # type: ignore[call-overload]

            described = str(command.get("description") or f"The {tool_name} {fn} command.")
            tools.append(
                _CreatureTool(
                    name=_tool_slug(tool_name, fn),
                    description=f"{tool_name}: {described}",
                    args_schema=args_schema,
                    action=action,
                    function=fn,
                    call=call,
                    loop=loop,
                    timeout=timeout,
                )
            )
    return tools


try:  # pragma: no cover - exercised only where crewai is installed
    from crewai.tools import BaseTool as _BaseTool
except Exception:  # noqa: BLE001 - the pure helpers above must import without crewai
    _BaseTool = object  # type: ignore[assignment,misc]


class _CreatureTool(_BaseTool):  # type: ignore[misc,valid-type]
    """One command of one Caspar tool, callable by an agent.

    The call goes out over the bridge to the tool's own creature, which decides
    what this project may do — the project id is never sent, because the tool
    reads it from the bearer token's topic instead. That is the whole
    authorization story, and it means a compromised runtime cannot reach another
    project's connected accounts by asking nicely.
    """

    action: str = ""
    function: str = ""
    call: Any = None
    loop: Any = None
    timeout: float = 120.0

    def _run(self, **kwargs: Any) -> str:
        payload = {"function": self.function, **{k: v for k, v in kwargs.items() if v not in ("", None)}}
        try:
            future = asyncio.run_coroutine_threadsafe(self.call(self.action, payload), self.loop)
            result = future.result(timeout=self.timeout)
        except Exception as exc:  # noqa: BLE001 - a tool failure is an answer
            logger.exception("tool %s failed", self.name)
            return f"Error: {self.name} could not be run: {exc or type(exc).__name__}"
        if isinstance(result, dict):
            if result.get("ok") is False:
                return f"Error: {result.get('error') or 'the tool refused the request'}"
            # The creature answers with its own fields; handing them back whole
            # lets the model read whatever the tool chose to report rather than
            # whatever shape this file happened to anticipate.
            return _clip(_render(result))
        return _clip(str(result))


def _render(result: dict[str, Any]) -> str:
    """A creature's reply, as something a model can read.

    JSON, minus the envelope fields that describe the call rather than answer
    it — an agent reading `"ok": true, "namespace": "github"` learns nothing and
    pays for the tokens.
    """
    import json

    body = {
        k: v
        for k, v in result.items()
        if k not in {"ok", "namespace", "action", "function", "correlationId"}
    }
    if not body:
        return "Done."
    try:
        return json.dumps(body, indent=2, default=str)
    except (TypeError, ValueError):
        return str(body)


# ── asking the people on the project ─────────────────────────────────────────

#: How long a run waits for a person to answer. Long enough for somebody to
#: notice and reply, short enough that a project nobody is watching finishes
#: instead of holding an agent — and its authorization — open indefinitely.
QUESTION_TIMEOUT_SECONDS = float(os.environ.get("DECILLION_QUESTION_TIMEOUT", "900"))

#: How long to wait for the platform to ACCEPT a question, as opposed to answer
#: it. Short, because this leg involves no person: the creature records the
#: question and says so. It exists to separate "nobody has answered yet" from
#: "the question never reached the project at all" — two situations that look
#: identical from inside an agent and want opposite responses.
QUESTION_ACCEPT_SECONDS = 45.0

#: How many questions one turn may ask. A run that asks endlessly is worse than
#: one that guesses: every question stops the work and costs somebody's
#: attention. Past this the tool tells the agent to decide for itself.
MAX_QUESTIONS_PER_RUN = 4


def ask_tool(
    call: Callable[[str, dict], Awaitable[dict]],
    await_result: Callable[[str, float], Awaitable[Any]],
    loop: asyncio.AbstractEventLoop,
    run_id: str,
    thread_id: str,
    agent_program_id: str,
    agent_name: str = "",
    ack: Callable[[str], None] | None = None,
) -> Any:
    """Let an agent put a question to the project and wait for the answer.

    Some work cannot be finished without a decision only a person can make —
    which direction to take, whether to publish, the detail nobody wrote down.
    An agent without this has two options and both are bad: guess, or stop and
    report that it could not proceed. It guesses.

    The waiting is the mechanism, not a complication of it. The call out to
    `crew/ask` is a request/response call like any other, and the creature
    simply does not answer it until a person has: the question is posted into
    the project's chat, and answering it publishes the result under the same
    correlation id (see `crew/answer`). So nothing polls, and nothing about a
    pending question lives in this sandbox — if the machine were replaced, the
    question would still be in the project's log where a person can see it.
    """
    from pydantic import BaseModel, Field

    class AskArgs(BaseModel):
        question: str = Field(
            description="The question to put to the people on this project. Ask one thing, "
            "plainly, and say what you will do with each answer."
        )
        options: str = Field(
            default="",
            description="Optional. A short list of answers to offer, separated by | — "
            "use it for a decision between known choices or a yes/no confirmation.",
        )

    class AskTheProject(_BaseTool):  # type: ignore[misc,valid-type]
        name: str = "ask_the_project"
        description: str = (
            "Ask the people on this project a question and wait for their answer. "
            "Use it when a decision is genuinely theirs — a choice between directions, "
            "a confirmation before something irreversible, or a fact you cannot find "
            "with your other tools. Your question appears in the project's chat and "
            "this waits for a reply, so ask only what you actually need."
        )
        args_schema: type[BaseModel] = AskArgs
        asked: int = 0

        def _run(self, question: str, options: str = "") -> str:
            if self.asked >= MAX_QUESTIONS_PER_RUN:
                return (
                    "You have already asked this project as much as one turn may ask. "
                    "Decide with what you have, and say in your answer which assumption "
                    "you made and why."
                )
            self.asked += 1
            choices = [o.strip() for o in str(options or "").split("|") if o.strip()]
            payload = {
                "question": str(question),
                "options": choices,
                "runId": run_id,
                "threadId": thread_id,
                "agentProgramId": agent_program_id,
                "agentName": agent_name,
            }
            return ask_question(
                lambda: asyncio.run_coroutine_threadsafe(
                    call("crew/ask", payload), loop
                ).result(timeout=QUESTION_ACCEPT_SECONDS),
                lambda answer_id: asyncio.run_coroutine_threadsafe(
                    await_result(answer_id, QUESTION_TIMEOUT_SECONDS), loop
                ).result(timeout=QUESTION_TIMEOUT_SECONDS + 30),
                ack,
            )

    return AskTheProject()


def ask_question(
    submit: Callable[[], Any],
    wait_for: Callable[[str], Any],
    ack: Callable[[str], None] | None = None,
) -> str:
    """Put a question to the project and report what came back.

    TWO waits, and the split is the point.
    The first is the platform ACCEPTING the question — no person is involved, so
    it is quick, and it either returns the id the answer will arrive under or it
    fails. The second is the person.

    Before this was split, a question that never reached the project at all was
    indistinguishable from one nobody had answered yet: both were a quarter of an
    hour of silence in the middle of a run, with nothing in the project to show a
    question had been asked. That is exactly what a missing route did — the
    gateway delivers an unrouted action to the grant's default handler, which
    records something and never replies.

    Every outcome is a STRING the agent can act on. A tool that raises here would
    end the turn; the point of asking is to carry on.
    """
    try:
        accepted = submit()
    except FuturesTimeout:
        return (
            "Error: this project did not accept the question — its runtime may not "
            "be able to reach the platform's question handler. Continue without "
            "asking, and say in your answer what you decided without confirmation."
        )
    except Exception as exc:  # noqa: BLE001 - a failed ask is an answer
        logger.exception("could not ask the project")
        return f"Error: the question could not be put to the project: {exc}"

    if not isinstance(accepted, dict) or accepted.get("ok") is False:
        reason = (accepted or {}).get("error") if isinstance(accepted, dict) else ""
        return f"Error: {reason or 'the question was refused by the project'}"
    answer_id = str(accepted.get("answerId") or "")
    if not answer_id:
        return (
            "Error: this project could not register the question. Continue without "
            "asking, and say what you decided without confirmation."
        )

    try:
        result = wait_for(answer_id)
    except FuturesTimeout:
        # Not an error: a project nobody is watching is an ordinary
        # situation, and the run should finish rather than hold its
        # authorization open until something else times it out.
        return (
            "Nobody answered in time. Continue with your best judgement, and say "
            "in your answer what you decided and that it was unconfirmed."
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("waiting for an answer failed")
        return f"Error: the answer never arrived: {exc}"
    if isinstance(result, dict):
        # Confirm receipt before doing anything with it. The platform holds an
        # answered question until the runtime says it arrived — because
        # publishing an answer to a bridge that had already gone used to retire
        # the question at the same time, losing both the answer and the only
        # record that anyone was still owed one.
        if ack is not None:
            question_id = str(result.get("questionId") or "")
            if question_id:
                try:
                    ack(question_id)
                except Exception:  # noqa: BLE001 - the answer still stands
                    logger.exception("could not acknowledge question %s", question_id)
        if result.get("ok") is False:
            return f"Error: {result.get('error') or 'the question was refused'}"
        answer = str(result.get("answer") or "").strip()
        if answer:
            return f"The project answered: {answer}"
    return "The project gave no answer. Continue with your best judgement."


# ── the CrewAI catalogue ─────────────────────────────────────────────────────


#: Built once. Which tools can be constructed is a property of the sandbox's
#: installed packages and environment, and neither changes between two prompts
#: of the same process — so paying the import-and-construct cost on every turn
#: would buy nothing.
_CATALOG_CACHE: list[Any] | None = None

#: Held while the catalogue is built. Building it INSTALLS packages, so two
#: turns arriving together must not both run pip against the same environment —
#: the second waits and takes the first one's result.
_CATALOG_LOCK = threading.Lock()


def catalog_tools() -> list[Any]:
    """Every `crewai_tools` tool that can actually be built in this sandbox.

    The catalogue is large and most of it needs a vendor account, so each tool
    is *constructed* and kept only if that worked. A tool needing an API key
    raises when its key is absent; one needing a file path or a database URI
    raises without it; both are exactly the tools an agent should not be shown.
    The set therefore grows by itself if an operator later puts a vendor key in
    the sandbox's environment, with nothing here to change.
    """
    global _CATALOG_CACHE
    if not _CATALOG_ENABLED:
        return []
    if _CATALOG_CACHE is not None:
        return list(_CATALOG_CACHE)
    with _CATALOG_LOCK:
        if _CATALOG_CACHE is not None:
            # Built while this call waited for the lock.
            return list(_CATALOG_CACHE)
        return _build_catalog()


def warm_catalog() -> int:
    """Build the catalogue now, off the hot path.

    Called once when the bridge comes up. Saying yes to the installers means the
    first build can take minutes, and doing that lazily would spend them inside
    somebody's first prompt — a project that looks hung at exactly the moment
    somebody is watching it. Blocking, so the caller decides which thread pays.
    """
    return len(catalog_tools())


def _build_catalog() -> list[Any]:
    """The catalogue itself. Call under `_CATALOG_LOCK`."""
    global _CATALOG_CACHE
    try:
        import crewai_tools
    except Exception:  # noqa: BLE001 - the catalogue is optional, the agent is not
        logger.info("crewai_tools is not installed; running with the project's own tools")
        _CATALOG_CACHE = []
        return []

    built: list[Any] = []
    # Several catalogue tools ASK before they work: their constructor prints
    # "You are missing the 'x' package. Would you like to install it? [y/N]:"
    # and reads a line. There is no terminal here, so the answer is given up
    # front — yes, every time. A sandbox is a disposable machine built to run
    # this project's agents, and the alternative to installing the package is
    # simply not having the tool.
    #
    # This is why the catalogue must be warmed BEFORE a prompt arrives (see
    # `warm_catalog`): saying yes means real installs, which take minutes the
    # first time a sandbox does it.
    with _auto_approve(), _installable():
        for name in sorted(getattr(crewai_tools, "__all__", []) or dir(crewai_tools)):
            if not name.endswith("Tool") or name.startswith("_"):
                continue
            candidate = getattr(crewai_tools, name, None)
            if not isinstance(candidate, type):
                continue
            try:
                instance = candidate()
            except BaseException:  # noqa: BLE001 - unbuildable means unusable
                logger.debug("skipping %s: it cannot be built without configuration", name)
                continue
            if not getattr(instance, "name", None) or not getattr(instance, "description", None):
                continue
            # A tool can build and still be unusable. Most of the catalogue is a
            # wrapper around somebody's API, and the framework says which key
            # each one needs — so a tool whose required key is not in this
            # sandbox is dropped here rather than offered and failed. Choosing
            # it would cost the agent a turn to learn what the platform already
            # knew, and the failure reads like the agent's mistake.
            if missing := _missing_env(instance):
                logger.debug("skipping %s: %s is not configured", name, ", ".join(missing))
                continue
            if unsupported := _unsupported_schema(instance):
                logger.warning(
                    "skipping %s: its arguments use %s, which a model provider refuses",
                    name,
                    ", ".join(unsupported),
                )
                continue
            built.append(instance)
    logger.info("catalogue tools available: %d", len(built))
    _CATALOG_CACHE = built
    return list(built)


#: JSON Schema keywords a model provider refuses in a function definition.
#: OpenAI answers `Invalid schema for function 'x': In context=(), 'allOf' is
#: not permitted` and fails the WHOLE request — every other tool included.
_UNSUPPORTED_SCHEMA_KEYWORDS = ("allOf",)


def _unsupported_schema(instance: Any) -> list[str]:
    """Schema keywords in this tool's arguments that a provider will refuse.

    A tool that builds but cannot be DESCRIBED is worse than one that fails to
    build. A model call carries every tool the agent has in one request, so a
    single unusable schema is answered with a 400 and the agent makes no tool
    calls at all — it does not lose one tool, it loses all of them, and then
    reports work it never did because the only thing left it can do is write an
    answer. One tool in the catalogue does exactly this today.

    So the schema is generated here and checked before the tool is offered. A
    schema that cannot be generated at all counts as unusable for the same
    reason.
    """
    schema_type = getattr(instance, "args_schema", None)
    if schema_type is None:
        return []
    try:
        schema = schema_type.model_json_schema()
    except Exception:  # noqa: BLE001 - undescribable is unusable
        return ["an unreadable schema"]
    found: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in _UNSUPPORTED_SCHEMA_KEYWORDS and key not in found:
                    found.append(key)
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(schema)
    return found


#: How many questions one constructor may be answered. A prompt loop that never
#: accepts the answer would otherwise spin forever on an endless "y"; this many
#: is far more than any real constructor asks, and the next read after it hits
#: EOF, which the constructor reports as a failure to build.
_MAX_APPROVALS = 64


@contextlib.contextmanager
def _auto_approve() -> Any:
    """Run a block whose prompts are answered yes, with its chatter discarded.

    Two problems, one substitution. A constructor that reads stdin would block
    forever here — there is no terminal in a sandbox, and a hung read on the
    first turn is invisible: no error, no log line, just a project whose agents
    never answer. And a constructor that is DECLINED gives up its tool, which is
    the whole reason the catalogue exists.

    So stdin is a fixed supply of "y". The bounded supply matters: unbounded
    approval turns a constructor that keeps asking into an infinite loop, while
    running out simply ends as EOF — a failure to build, which is handled.
    """
    saved_in, saved_out = sys.stdin, sys.stdout
    sys.stdin = io.StringIO("y\n" * _MAX_APPROVALS)
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdin, sys.stdout = saved_in, saved_out


#: A throwaway uv project the catalogue's installers can add packages to.
#: Inside the runtime's own directory, so it is destroyed with the machine and
#: never touches the cloned repository.
_INSTALL_PROJECT = Path(os.environ.get("CREWAI_HOME", "/opt/crewai")) / ".decillion-tools"


@contextlib.contextmanager
def _installable() -> Any:
    """Make the catalogue's `uv add` calls actually work.

    Saying yes to an installer is only half of it. The tools install with
    `uv add <package>`, which needs two things this process does not have: a uv
    PROJECT in the working directory (without one it fails with "No
    pyproject.toml found" and the package is never installed), and somewhere to
    install to — by default a `.venv` beside that project, which is not the
    interpreter running this code.

    So a scratch project is created to be the thing `uv add` edits, and uv is
    pointed at the venv we are actually running in. The scratch project is
    disposable and inside the runtime's own directory: the cloned repository is
    never modified, and a re-provisioned machine starts clean.

    The working directory is process-global, which is why this is held only
    around the catalogue build (under `_CATALOG_LOCK`) and restored afterwards.
    Every path this module uses elsewhere is absolute for the same reason.
    """
    saved_cwd = os.getcwd()
    saved_env = os.environ.get("UV_PROJECT_ENVIRONMENT")
    try:
        _INSTALL_PROJECT.mkdir(parents=True, exist_ok=True)
        manifest = _INSTALL_PROJECT / "pyproject.toml"
        if not manifest.exists():
            manifest.write_text(
                '[project]\nname = "decillion-tools"\nversion = "0.0.0"\n'
                'requires-python = ">=3.10"\ndependencies = []\n',
                encoding="utf-8",
            )
        os.environ["UV_PROJECT_ENVIRONMENT"] = sys.prefix
        os.chdir(_INSTALL_PROJECT)
    except OSError:
        # No scratch project means the installers fail as they did before —
        # those tools are skipped, and every tool that needs no install is
        # unaffected. Not worth losing the catalogue over.
        logger.debug("could not prepare an install target for the tool catalogue")
    try:
        yield
    finally:
        os.chdir(saved_cwd)
        if saved_env is None:
            os.environ.pop("UV_PROJECT_ENVIRONMENT", None)
        else:
            os.environ["UV_PROJECT_ENVIRONMENT"] = saved_env


def _missing_env(instance: Any) -> list[str]:
    """The required environment variables this tool declares and does not have.

    Read off the tool itself (`env_vars`), because the framework already carries
    the answer — there is no list here to keep in step with a catalogue of a
    hundred tools that grows every release.
    """
    missing: list[str] = []
    for var in getattr(instance, "env_vars", None) or []:
        if isinstance(var, dict):
            name, required = str(var.get("name") or ""), bool(var.get("required"))
        else:
            name, required = str(getattr(var, "name", "") or ""), bool(getattr(var, "required", False))
        if name and required and not os.environ.get(name):
            missing.append(name)
    return missing


# ── what one turn is given ───────────────────────────────────────────────────


def build_tools(
    tool_specs: list[dict[str, Any]],
    call: Callable[[str, dict], Awaitable[dict]] | None,
    loop: asyncio.AbstractEventLoop | None,
    turn: dict[str, str] | None = None,
    await_result: Callable[[str, float], Awaitable[Any]] | None = None,
    ack_question: Callable[[str], None] | None = None,
) -> list[Any]:
    """Everything the agents on this turn can use.

    Ordered deliberately: the project's own machine first, then the tools the
    project has attached, then asking the people on it, then the general
    catalogue. A model reads the list in order, and the first ones are those
    that act on THIS project.

    `turn` identifies the run for the tools that need to say who is speaking —
    a question has to be attributed to an agent and tied to the run waiting on
    it, which is not something the tool can find out for itself.
    """
    tools: list[Any] = []
    try:
        tools.extend(workspace_tools())
    except Exception:  # noqa: BLE001 - a turn without file tools still runs
        logger.exception("could not build the project's workspace tools")
    if call is not None and loop is not None:
        try:
            tools.extend(creature_tools(tool_specs, call, loop))
        except Exception:  # noqa: BLE001
            logger.exception("could not build the project's Caspar tools")
        if turn and await_result is not None:
            try:
                tools.append(
                    ask_tool(
                        call,
                        await_result,
                        loop,
                        turn.get("runId", ""),
                        turn.get("threadId", "main"),
                        turn.get("agentProgramId", ""),
                        turn.get("agentName", ""),
                        ack_question,
                    )
                )
            except Exception:  # noqa: BLE001
                logger.exception("could not build the project's question tool")
    try:
        tools.extend(catalog_tools())
    except Exception:  # noqa: BLE001
        logger.exception("could not build the CrewAI tool catalogue")
    return tools
