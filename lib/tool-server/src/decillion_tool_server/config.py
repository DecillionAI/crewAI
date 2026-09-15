"""What this tool server is, read from the sandbox it runs in.

Everything here is written into `/etc/decillion/bridge.env` by the same call
that created the sandbox (`spaces/create`), which is the only place the
project's bearer token ever appears. Nothing is passed on a command line, so
the token is not visible in the sandbox's process list.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class BridgeConfig:
    """The identity and endpoints one tool-server process runs with.

    Note what is NOT here any more: a model-proxy port. This process makes no
    model calls, `llm/chat` is not in its grant's routes, and there is nothing
    for a provider key to leak through.
    """

    gateway_url: str
    space_id: str
    topic: str
    token: str
    crew_home: str
    log_level: str
    #: Where this process keeps what it has not yet delivered. On the project's
    #: own volume, so an undelivered tool result survives the machine sleeping.
    state_dir: str
    #: The pinned revision of the tool catalogue installed on this volume,
    #: reported on announce so the node can tell an intended install from a
    #: finished one.
    runtime_ref: str

    @property
    def configured(self) -> bool:
        return bool(self.gateway_url and self.space_id and self.token)

    def describe(self) -> str:
        """A one-line summary safe to log — never includes the token."""
        return (
            f"space={self.space_id} topic={self.topic} "
            f"gateway={self.gateway_url} token={'set' if self.token else 'MISSING'}"
        )


def load_config() -> BridgeConfig:
    """Read the bridge's configuration from the environment.

    The topic defaults to `space:<id>`, which is how `spaces/create` names it;
    it is still read from the environment so the two can diverge without a
    code change if the platform ever scopes a bridge differently.
    """
    space_id = os.environ.get("DECILLION_SPACE_ID", "").strip()
    return BridgeConfig(
        gateway_url=os.environ.get("CASPAR_GATEWAY_URL", "").strip(),
        space_id=space_id,
        topic=os.environ.get("DECILLION_BRIDGE_TOPIC", "").strip()
        or (f"space:{space_id}" if space_id else ""),
        token=os.environ.get("DECILLION_BRIDGE_TOKEN", "").strip(),
        crew_home=os.environ.get("CREWAI_HOME", "/opt/crewai").strip(),
        log_level=os.environ.get("DECILLION_LOG_LEVEL", "INFO").strip().upper(),
        state_dir=os.environ.get("DECILLION_STATE_DIR", "/data/.decillion/state").strip(),
        runtime_ref=os.environ.get("DECILLION_RUNTIME_REF", "").strip(),
    )
