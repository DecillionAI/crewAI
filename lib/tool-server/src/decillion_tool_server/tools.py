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
    """Read, write and run things on the project's own machine.

    An import failure here is degraded, not fatal. These are built on
    `crewai.tools.BaseTool`, and if that import fails — a broken install, a
    partial upgrade — the machine can still say it is up and say it has nothing
    to offer. It used to raise instead, out of `announce()`, which runs on every
    connect: the connection died on its own success path, reconnected, and died
    again. A project sat at "Starting the tool server" forever with no tools and
    no error, which is the one outcome worse than having no catalogue.
    """
    try:
        from crewai.tools import BaseTool
        from pydantic import BaseModel, Field
    except ImportError as exc:
        logger.error(
            "this machine can offer no tools: %s. The tool server is running, "
            "but crewai is not importable in its environment.",
            exc,
        )
        return []

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


try:  # pragma: no cover - exercised only where crewai is installed
    from crewai.tools import BaseTool as _BaseTool
except Exception:  # noqa: BLE001 - the pure helpers above must import without crewai
    _BaseTool = object  # type: ignore[assignment,misc]


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


# ── The server's entry points ────────────────────────────────────────────────
#
# The node addresses a tool by NAME and hands it a dict of arguments. It does
# not know whether that name is a workspace tool or something out of the
# catalogue, and it should not: which tools exist on a given machine depends on
# what installed successfully there, which is exactly what `tool_manifest`
# reports back on announce.


def _all_tools() -> dict[str, Any]:
    """Every tool this machine can actually run, by name.

    The project's OWN tools come first and win a name collision. A catalogue
    tool that happens to be called `FileReadTool` must never shadow the one that
    reads this project's files — the agent asked about this project.
    """
    out: dict[str, Any] = {}
    for tool in catalog_tools():
        name = getattr(tool, "name", "")
        if name:
            out[name] = tool
    for tool in workspace_tools():
        name = getattr(tool, "name", "")
        if name:
            out[name] = tool
    return out


def tool_manifest() -> list[dict[str, Any]]:
    """What this machine can run, as the node needs to describe it to a model.

    Sent on announce so the node can offer these tools to an agent without
    asking first — a round trip on the critical path of every first prompt.
    """
    manifest: list[dict[str, Any]] = []
    for name, tool in _all_tools().items():
        entry: dict[str, Any] = {
            "name": name,
            "description": str(getattr(tool, "description", "") or ""),
            "kind": "sandbox",
        }
        schema = getattr(tool, "args_schema", None)
        if schema is not None:
            try:
                entry["parameters"] = schema.model_json_schema()
            except Exception:  # noqa: BLE001 - a tool with an unreadable schema still runs
                entry["parameters"] = {"type": "object", "properties": {}}
        else:
            entry["parameters"] = {"type": "object", "properties": {}}
        manifest.append(entry)
    return manifest


def build_catalog() -> list[Any]:
    """Build the catalogue now. Blocking, so the caller decides which thread pays."""
    return catalog_tools()


def run_tool(name: str, args: dict[str, Any]) -> str:
    """Run one tool and return what it said.

    Synchronous and blocking — a catalogue tool is ordinary Python and most of
    them make network calls. The caller runs it on a worker thread.

    A tool that does not exist is an ANSWER, not an exception: the model chose a
    name, and being told which names are real is something it can act on. It
    happens most on a machine where an optional tool failed to install, and the
    difference between "no such tool" and a stack trace is the difference
    between the agent recovering and the turn ending.
    """
    tools = _all_tools()
    tool = tools.get(name)
    if tool is None:
        available = ", ".join(sorted(tools)) or "none"
        return f"Error: this project's machine has no tool called {name!r}. It has: {available}"
    try:
        result = tool.run(**(args or {}))
    except TypeError as exc:
        # Wrong arguments: the model can fix this on the next turn if it is told
        # what the tool wanted.
        schema = getattr(tool, "args_schema", None)
        wanted = ""
        if schema is not None:
            try:
                wanted = ", ".join((schema.model_json_schema().get("properties") or {}).keys())
            except Exception:  # noqa: BLE001
                wanted = ""
        detail = f" It takes: {wanted}." if wanted else ""
        return f"Error: {name} was called with the wrong arguments ({exc}).{detail}"
    return _clip(result if isinstance(result, str) else str(result))
