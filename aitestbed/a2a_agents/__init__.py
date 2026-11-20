"""Local A2A (Agent2Agent protocol) agents for the 6G AI Traffic Testbed.

These are minimal, deterministic A2A agent servers used to generate
reproducible agent-to-agent traffic on the loopback interface (so it can be
shaped by tc/netem and captured by tcpdump exactly like MCP-over-HTTP). They
intentionally do **not** call an LLM, so the local A2A scenarios need no API
keys and produce a stable traffic shape for SA4 cross-checks.

Modules:
  - ``common``              shared card builders + uvicorn bootstrap (imports a2a-sdk)
  - ``echo_agent``          a leaf agent: ``python -m a2a_agents.echo_agent --port N``
  - ``orchestrator_agent``  an agent that delegates to downstream A2A agents
  - ``launcher``            subprocess manager used by the scenario (no a2a-sdk import)

Only ``launcher`` is safe to import without ``a2a-sdk`` installed; the agent
modules import the SDK and are normally run as subprocesses.
"""
