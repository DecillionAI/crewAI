"""Test fixtures for the bridge.

Every test gets its own durable-state directory. The bridge persists its outbox
and its work record to `/data/.decillion` inside a sandbox, and a suite that
shared one directory would leak a finished run from one test into the work
history of the next — and would write into a real volume on a machine that has
one.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def isolated_state_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("DECILLION_STATE_DIR", str(tmp_path / "state"))
    yield
