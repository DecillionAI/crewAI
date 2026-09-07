"""The Caspar bridge: a Decillion project's agents, running as a CrewAI crew.

One bridge process runs inside each project's Modal sandbox. It holds a single
connection to the Caspar node, subscribed to that project's topic, and is the
only thing on either side that knows about both worlds:

    Caspar creature  ── publishUpdate ──►  bridge  ──►  CrewAI crew
    CrewAI crew      ──►  bridge  ── /gateway/signal ──►  Caspar creature

It authenticates with a bearer token the project's `spaces/create` minted and
wrote into this sandbox — the bridge has no Caspar key, and the token is scoped
to one topic, so a bridge can only ever speak for its own project.
"""

__version__ = "0.1.0"
