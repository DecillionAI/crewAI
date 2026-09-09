"""Where this bridge keeps what must survive it.

A project's sandbox is not permanent: it sleeps after five idle minutes and the
runtime terminates it, and a bridge crash restarts the process. What survives
either of those is the Modal Volume mounted at `/data` — the project's own disk.
So anything the platform would otherwise lose when this process ends belongs in
a file under here, not in a Python object.

Two things do:

* the **outbox** — run events that have been produced but not yet accepted by
  the node (see `outbox.py`), and
* the **work record** — what each run was and how it ended, which `crew/work`
  answers from and which used to live only in a bounded in-memory deque.

`/data` is the default because that is the volume; the directory is overridable
so the runtime can be exercised on a machine that has no such mount.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_STATE_DIR = "/data/.decillion"


def state_dir(name: str = "") -> Path:
    """The bridge's durable directory, created on first use.

    A machine where `/data` is not writable (a unit test, a developer's laptop)
    falls back to a temporary directory rather than failing: losing durability
    off the platform is fine, refusing to start is not.
    """
    root = Path(os.environ.get("DECILLION_STATE_DIR", _DEFAULT_STATE_DIR))
    target = root / name if name else root
    try:
        target.mkdir(parents=True, exist_ok=True)
        return target
    except OSError:
        fallback = Path(tempfile.gettempdir()) / "decillion-bridge"
        target = fallback / name if name else fallback
        target.mkdir(parents=True, exist_ok=True)
        logger.warning("durable state falls back to %s", target)
        return target


def write_json_atomically(path: Path, value: Any) -> bool:
    """Write one JSON document so a crash mid-write cannot corrupt it.

    Temp file in the same directory, then rename — the rename is atomic on the
    filesystem, so a reader sees either the previous document or the new one and
    never half of either.
    """
    try:
        with tempfile.NamedTemporaryFile(
            "w", dir=str(path.parent), delete=False, encoding="utf-8"
        ) as handle:
            json.dump(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
            temp = handle.name
        os.replace(temp, path)
        return True
    except (OSError, TypeError, ValueError):
        logger.exception("could not persist %s", path.name)
        return False


def read_json(path: Path) -> Any:
    """Read one JSON document, or None if it is missing or unreadable.

    A truncated file is discarded rather than raised: the caller is recovering
    from a crash, and one damaged record must not stop the rest from replaying.
    """
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        logger.warning("discarding unreadable state file %s", path.name)
        return None
