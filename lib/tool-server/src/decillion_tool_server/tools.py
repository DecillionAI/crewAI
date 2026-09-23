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
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import TimeoutError as FuturesTimeout
from pathlib import Path
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

#: The project's own directory on its machine. This is the sandbox's persistent
#: volume and the root `spaces/files` reads, so it is the one place where an
#: agent's work is both visible to the project and durable across a sleep.
#: Anything written elsewhere in the container is lost when the machine stops.
WORKSPACE_ROOT = os.environ.get("DECILLION_WORKSPACE", "/data")


def _shell_env() -> dict[str, str]:
    """GUI apps an agent starts land on Computer when that session is up."""
    env = os.environ.copy()
    root = Path(os.environ.get("DECILLION_WORKSPACE", WORKSPACE_ROOT))
    env_file = root / ".autobot" / "desktop.env"
    ready = root / ".autobot" / "desktop-ready"
    if env_file.is_file():
        try:
            for raw in env_file.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                env[key.strip()] = value.strip().strip('"').strip("'")
        except OSError:
            logger.debug("could not read desktop.env")
    elif ready.is_file():
        env.setdefault("DISPLAY", ":1")
    return env


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


def _ensure_desktop_session() -> str:
    """Start or join the project's shared graphical session.

    The same session the person sees when they open Computer: one DISPLAY, one
    x11vnc password, /data as the home of the file manager. Lazily started so
    idle projects pay nothing for X.

    Agents normally reach Computer through the platform (`ensure_computer` is
    intercepted and runs the same startDesktop path as opening Computer on the
    orbit). This local path still joins or re-ensures when the start script is
    already on the volume.
    """
    root = Path(WORKSPACE_ROOT)
    autobot = root / ".autobot"
    ready = autobot / "desktop-ready"
    starter = autobot / "desktop-start.sh"
    autobot.mkdir(parents=True, exist_ok=True)

    if ready.is_file():
        # Confirm the proxy is still answering; a stale marker used to claim a
        # dead session was live.
        probe = subprocess.run(  # noqa: S603
            [
                "python3",
                "-c",
                "import http.client,sys;"
                "c=http.client.HTTPConnection('127.0.0.1',6080,timeout=2);"
                "c.request('GET','/vnc.html');"
                "b=c.getresponse().read(200).lower();"
                "sys.exit(0 if (b'html' in b or b'novnc' in b) else 1)",
            ],
            capture_output=True,
            timeout=8,
        )
        if probe.returncode == 0:
            return (
                "Computer is already on (DISPLAY=:1). File manager, browser and "
                "terminal share /data with Files. GUI apps you start appear there; "
                "the person sees the same session when they open Computer."
            )

    if starter.is_file():
        try:
            completed = subprocess.run(  # noqa: S603
                ["sh", str(starter), "ensure"],
                cwd=WORKSPACE_ROOT,
                env=_shell_env(),
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=90,
                start_new_session=True,
            )
        except subprocess.TimeoutExpired:
            return (
                "Computer is still starting. Wait briefly and call ensure_computer "
                "again — or ask the person to open Computer on the orbit to join "
                "this same session once the address appears."
            )
        except OSError as exc:
            return f"Error: could not start Computer: {exc}"
        if ready.is_file() or completed.returncode == 0:
            return (
                "Computer is on (DISPLAY=:1). File manager opens /data (same as Files). "
                "Launch firefox-esr or a terminal with DISPLAY set — they appear on "
                "the shared desktop the person can open from the orbit."
            )
        tail = (completed.stderr or completed.stdout or "").strip()[-400:]
        return f"Error: Computer did not come up.{(' ' + tail) if tail else ''}"

    # No start script yet: the platform should have intercepted ensure_computer
    # and written one via startDesktop. If we still see this, ask for a retry
    # rather than telling the agent only a person can start Computer.
    return (
        "Computer is not initialized on this machine yet. Call ensure_computer "
        "again so the platform can start the shared session (same as opening "
        "Computer on the orbit). If that still fails, ask the person to open "
        "Computer once, then continue."
    )


_COMPUTER_HINT = (
    "Use open_on_computer to open a URL, then computer_click / computer_move / "
    "computer_type / computer_key for the mouse and keyboard. Do not launch "
    "chromium yourself — wrong flags crash the shared desktop."
)


def _computer_is_on() -> bool:
    return (Path(WORKSPACE_ROOT) / ".autobot" / "desktop-ready").is_file()


def _computer_off_message() -> str:
    return (
        "Computer is not on yet. Call ensure_computer first, wait until it says "
        f"the session is up, then retry. {_COMPUTER_HINT}"
    )


def _xdotool(*args: str) -> str:
    """Drive the shared Computer pointer/keyboard. DISPLAY comes from desktop.env."""
    if not _computer_is_on():
        return _computer_off_message()
    binary = shutil.which("xdotool")
    if not binary:
        return (
            "xdotool is not on this machine yet (Computer fetches it in the background "
            "on first start). Wait about a minute and retry, or ask the person to open "
            "Computer once so the desktop finishes installing helpers."
        )
    try:
        completed = subprocess.run(  # noqa: S603
            [binary, *args],
            env=_shell_env(),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        return "Error: the Computer input command timed out"
    except OSError as exc:
        return f"Error: could not run xdotool: {exc}"
    parts = []
    if completed.stdout.strip():
        parts.append(completed.stdout.strip())
    if completed.stderr.strip():
        parts.append(f"[stderr] {completed.stderr.strip()}")
    if completed.returncode != 0:
        parts.append(f"[exit code {completed.returncode}]")
    return "\n".join(parts) if parts else "ok"


def _display_size() -> tuple[int, int] | None:
    binary = shutil.which("xdotool")
    if not binary or not _computer_is_on():
        return None
    try:
        completed = subprocess.run(  # noqa: S603
            [binary, "getdisplaygeometry"],
            env=_shell_env(),
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    parts = (completed.stdout or "").strip().split()
    if len(parts) < 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def _open_on_computer(url: str) -> str:
    """Open a URL in the project's browser with the flags that keep X alive."""
    if not _computer_is_on():
        return _computer_off_message()
    target = (url or "").strip() or "about:blank"
    if not re.match(r"^(https?://|about:)", target, re.I):
        target = "https://" + target.lstrip("/")
    launcher = Path(WORKSPACE_ROOT) / ".autobot" / "desktop-launch.sh"
    env = _shell_env()
    if launcher.is_file():
        try:
            subprocess.Popen(  # noqa: S603
                ["sh", str(launcher), "browser", target],
                cwd=WORKSPACE_ROOT,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError as exc:
            return f"Error: could not open the browser: {exc}"
    else:
        # Older sessions without an updated launcher: still prefer safe Chromium flags.
        browser = (
            shutil.which("chromium")
            or shutil.which("chromium-browser")
            or shutil.which("google-chrome")
            or shutil.which("firefox-esr")
            or shutil.which("firefox")
        )
        if not browser:
            return "Error: no browser is installed on this machine yet."
        cmd = [browser]
        name = Path(browser).name
        if "chrom" in name or "chrome" in name:
            cmd.extend(
                [
                    "--user-data-dir=/tmp/decillion-browser",
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-gpu",
                    "--no-first-run",
                    "--no-default-browser-check",
                ]
            )
        cmd.append(target)
        try:
            subprocess.Popen(  # noqa: S603
                cmd,
                cwd=WORKSPACE_ROOT,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError as exc:
            return f"Error: could not open the browser: {exc}"
    size = _display_size()
    geometry = f" Screen is {size[0]}x{size[1]}." if size else ""
    return (
        f"Opened {target} on Computer (DISPLAY=:1).{geometry} "
        f"Wait a few seconds for the page to load, then use computer_click / "
        f"computer_move / computer_type. {_COMPUTER_HINT}"
    )


def _computer_screenshot(path: str = "") -> str:
    if not _computer_is_on():
        return _computer_off_message()
    dest = (path or "").strip() or f"/tmp/computer-{int(time.time())}.png"
    if dest.startswith("/tmp/"):
        out = Path(dest)
    else:
        try:
            out = _resolve(dest)
        except ValueError as exc:
            return f"Error: {exc}"
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return f"Error: could not create folder for screenshot: {exc}"
    env = _shell_env()
    attempts: list[list[str]] = []
    scrot = shutil.which("scrot")
    if scrot:
        attempts.append([scrot, "-o", str(out)])
    imagemagick = shutil.which("import")
    if imagemagick:
        attempts.append([imagemagick, "-window", "root", str(out)])
    if not attempts:
        return (
            "No screenshot tool is installed yet (Computer fetches scrot in the background). "
            "Wait a minute and retry."
        )
    last_err = ""
    for cmd in attempts:
        try:
            completed = subprocess.run(  # noqa: S603
                cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            last_err = str(exc)
            continue
        if completed.returncode == 0 and out.is_file():
            label = str(out) if dest.startswith("/tmp/") else dest
            return f"Saved screenshot to {label} ({out.stat().st_size} bytes)."
        last_err = (completed.stderr or completed.stdout or f"exit {completed.returncode}").strip()
    return f"Error: screenshot failed. {last_err}".strip()


def workspace_tools() -> list[Any]:
    """Read, write and run things on the project's own machine.

    These deliberately do not inherit CrewAI's BaseTool. Agents no longer run
    in this process, and importing the entire framework just to obtain a tiny
    ``run -> _run`` adapter put its dependency graph on every cold boot's
    critical path. The bridge contract is structural: name, description,
    schema and ``run(**args)``.
    """
    try:
        from pydantic import BaseModel, Field
    except ImportError as exc:
        logger.error(
            "this machine can offer no workspace tools: %s",
            exc,
        )
        return []

    class WorkspaceTool:
        args_schema: type[BaseModel]

        def run(self, **kwargs: Any) -> Any:
            values = self.args_schema.model_validate(kwargs).model_dump()
            return self._run(**values)

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

    class ReadFile(WorkspaceTool):
        name = "read_project_file"
        description = (
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

    class WriteFile(WorkspaceTool):
        name = "write_project_file"
        description = (
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

    class AppendFile(WorkspaceTool):
        name = "append_project_file"
        description = (
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

    class ListFiles(WorkspaceTool):
        name = "list_project_files"
        description = (
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

    class RunShell(WorkspaceTool):
        name = "run_shell_command"
        description = (
            "Run a shell command on the project's machine, in the project's folder (/data). "
            "That folder is the same tree Files and the Computer file manager show. "
            "For the shared desktop: call ensure_computer, then open_on_computer / "
            "computer_click / computer_move — do not launch chromium yourself (wrong "
            "flags crash Computer). For a site that needs a human login, ask the person "
            "to open Computer and complete it. Returns the command's output; a command "
            "that takes longer than five minutes is stopped."
        )
        args_schema: type[BaseModel] = ShellArgs

        def _run(self, command: str) -> str:
            try:
                completed = subprocess.run(  # noqa: S602 - a shell is the point
                    str(command),
                    shell=True,
                    cwd=WORKSPACE_ROOT,
                    env=_shell_env(),
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    text=True,
                    timeout=SHELL_TIMEOUT_SECONDS,
                    start_new_session=True,
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

    class EnsureComputerArgs(BaseModel):
        pass

    class EnsureComputer(WorkspaceTool):
        name = "ensure_computer"
        description = (
            "Start the project's shared graphical Computer if it is not already on: "
            "a desktop with file manager on /data (same as Files), browser and terminal. "
            "Agents may start it themselves; the person opens Computer on the orbit to "
            "watch or take control of the same session. Call this before open_on_computer "
            "or computer_click. Idle projects keep Computer off to save memory."
        )
        args_schema: type[BaseModel] = EnsureComputerArgs

        def _run(self) -> str:
            msg = _ensure_desktop_session()
            if "Computer is on" in msg or "already on" in msg:
                return f"{msg} {_COMPUTER_HINT}"
            return msg

    class OpenOnComputerArgs(BaseModel):
        url: str = Field(description="URL to open in the shared Computer browser (https://…)")

    class OpenOnComputer(WorkspaceTool):
        name = "open_on_computer"
        description = (
            "Open a URL in the project's shared Computer browser (safe Chromium/Firefox "
            "flags). Call ensure_computer first. Then use computer_click / computer_move "
            "to interact. Do not use run_shell_command to launch a browser."
        )
        args_schema: type[BaseModel] = OpenOnComputerArgs

        def _run(self, url: str) -> str:
            return _open_on_computer(url)

    class ComputerClickArgs(BaseModel):
        x: int = Field(description="Horizontal pixel position on the Computer screen")
        y: int = Field(description="Vertical pixel position on the Computer screen")
        button: int = Field(default=1, description="Mouse button: 1=left, 2=middle, 3=right")

    class ComputerClick(WorkspaceTool):
        name = "computer_click"
        description = (
            "Click at a pixel on the shared Computer screen (DISPLAY=:1). "
            "Call ensure_computer and open_on_computer first. Bottom-left of an "
            "WxH screen is near (5, H-5)."
        )
        args_schema: type[BaseModel] = ComputerClickArgs

        def _run(self, x: int, y: int, button: int = 1) -> str:
            result = _xdotool("mousemove", str(int(x)), str(int(y)), "click", str(int(button) or 1))
            if result.startswith("Error") or "not on" in result or "not on this" in result:
                return result
            return f"Clicked ({int(x)}, {int(y)}) button {int(button) or 1}. {result}".strip()

    class ComputerMoveArgs(BaseModel):
        x: int = Field(description="Horizontal pixel position on the Computer screen")
        y: int = Field(description="Vertical pixel position on the Computer screen")

    class ComputerMove(WorkspaceTool):
        name = "computer_move"
        description = (
            "Move the mouse pointer on the shared Computer without clicking. "
            "Bottom-left of an WxH screen is near (5, H-5)."
        )
        args_schema: type[BaseModel] = ComputerMoveArgs

        def _run(self, x: int, y: int) -> str:
            result = _xdotool("mousemove", str(int(x)), str(int(y)))
            if result.startswith("Error") or "not on" in result or "xdotool is not" in result:
                return result
            return f"Moved pointer to ({int(x)}, {int(y)}). {result}".strip()

    class ComputerTypeArgs(BaseModel):
        text: str = Field(description="Text to type into the focused window on Computer")

    class ComputerType(WorkspaceTool):
        name = "computer_type"
        description = "Type text into the focused window on the shared Computer."
        args_schema: type[BaseModel] = ComputerTypeArgs

        def _run(self, text: str) -> str:
            result = _xdotool("type", "--clearmodifiers", "--", str(text))
            if result.startswith("Error") or "not on" in result or "xdotool is not" in result:
                return result
            return f"Typed {len(str(text))} characters. {result}".strip()

    class ComputerKeyArgs(BaseModel):
        key: str = Field(
            description="Key name for xdotool (Return, Tab, Escape, ctrl+a, …)"
        )

    class ComputerKey(WorkspaceTool):
        name = "computer_key"
        description = "Press a key or key combo on the shared Computer (Return, Tab, ctrl+l, …)."
        args_schema: type[BaseModel] = ComputerKeyArgs

        def _run(self, key: str) -> str:
            result = _xdotool("key", "--clearmodifiers", str(key))
            if result.startswith("Error") or "not on" in result or "xdotool is not" in result:
                return result
            return f"Pressed {key}. {result}".strip()

    class ComputerScreenshotArgs(BaseModel):
        path: str = Field(
            default="",
            description="Optional path under the project (or /tmp/…). Empty = /tmp/computer-….png",
        )

    class ComputerScreenshot(WorkspaceTool):
        name = "computer_screenshot"
        description = (
            "Capture the shared Computer screen to a PNG. Use this to verify a page "
            "loaded or a click landed before claiming the browser task succeeded."
        )
        args_schema: type[BaseModel] = ComputerScreenshotArgs

        def _run(self, path: str = "") -> str:
            return _computer_screenshot(path)

    return [
        ReadFile(),
        WriteFile(),
        AppendFile(),
        ListFiles(),
        RunShell(),
        EnsureComputer(),
        OpenOnComputer(),
        ComputerClick(),
        ComputerMove(),
        ComputerType(),
        ComputerKey(),
        ComputerScreenshot(),
    ]


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


def catalog_tools(*, build: bool = True) -> list[Any]:
    """Every `crewai_tools` tool that can actually be built in this sandbox.

    `build=False` answers with what is already built and never starts a build.
    Building is not a cheap read: it constructs the whole catalogue and says YES
    to the package installs some constructors ask for, which is minutes of
    network on a cold machine. Anything on a latency path — above all announcing
    this machine to the node — must take that door.

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
    if not build:
        return []
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


#: Catalogue tools that look buildable but belong to a different product. Offering
#: them burns a turn: the agent picks Daytona (or similar), learns the package is
#: missing, and never uses the project's own Computer tools that actually work.
_CATALOG_BLOCKLIST = frozenset(
    {
        "DaytonaExecTool",
        "DaytonaFileTool",
        "DaytonaPythonTool",
        "DaytonaBaseTool",
    }
)


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
            if name in _CATALOG_BLOCKLIST or name.startswith("Daytona"):
                logger.debug("skipping %s: foreign sandbox tool, not this project's machine", name)
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


def _all_tools(*, build: bool = True) -> dict[str, Any]:
    """Every tool this machine can actually run, by name.

    The project's OWN tools come first and win a name collision. A catalogue
    tool that happens to be called `FileReadTool` must never shadow the one that
    reads this project's files — the agent asked about this project.
    """
    out: dict[str, Any] = {}
    for tool in catalog_tools(build=build):
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
    # Never a build: this is what an ANNOUNCE carries, and an announcement that
    # waited for the catalogue was an announcement that never happened. The
    # workspace tools need no install and are always here; the catalogue joins
    # them when it is warm, and the node is told again then.
    for name, tool in _all_tools(build=False).items():
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
    except (TypeError, ValueError) as exc:
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
