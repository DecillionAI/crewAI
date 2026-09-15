"""A call to a creature waits for the creature's answer, not for delivery."""

import asyncio

import pytest

from decillion_tool_server.client import CasparBridgeClient
from decillion_tool_server.config import BridgeConfig


def _client():
    cfg = BridgeConfig(
        gateway_url="ws://127.0.0.1:1",
        space_id="s1",
        topic="space:s1",
        token="t" * 40,
        crew_home="/opt/crewai",
        log_level="INFO",
        state_dir="/tmp/decillion-tool-server-test",
        runtime_ref="abc123",
    )

    async def on_update(key, data):
        return None

    return CasparBridgeClient(cfg, on_update)

