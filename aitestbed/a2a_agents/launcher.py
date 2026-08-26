"""Subprocess manager for local A2A agents.

Spawns the agent entry-point modules as child processes and waits until each
agent's card endpoint is serving. Deliberately imports neither a2a-sdk nor any
scenario code (only stdlib + httpx) so ``import scenarios`` stays cheap and
does not require a2a-sdk to be installed.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import httpx

# aitestbed package root, so `python -m a2a_agents.<mod>` resolves in the child.
_PKG_ROOT = Path(__file__).resolve().parent.parent

AGENT_CARD_PATH = "/.well-known/agent-card.json"


class A2AAgentProcess:
    """A single local A2A agent running as a subprocess on a fixed port."""

    def __init__(self, module: str, args: list[str], port: int, host: str = "127.0.0.1"):
        self.module = module
        self.args = args
        self.port = port
        self.host = host
        self._proc: Optional[subprocess.Popen] = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self) -> None:
        if self._proc is not None:
            return
        cmd = [sys.executable, "-m", self.module, "--port", str(self.port),
               "--host", self.host, *self.args]
        self._proc = subprocess.Popen(
            cmd,
            cwd=str(_PKG_ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def wait_ready(self, timeout: float = 20.0) -> bool:
        """Poll the agent card endpoint until it serves or *timeout* elapses."""
        deadline = time.time() + timeout
        url = self.base_url + AGENT_CARD_PATH
        while time.time() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                return False  # process exited early
            try:
                r = httpx.get(url, timeout=2.0)
                if r.status_code == 200:
                    return True
            except Exception:
                pass
            time.sleep(0.25)
        return False

    def stop(self) -> None:
        if self._proc is None:
            return
        try:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        except Exception:
            pass
        finally:
            self._proc = None


class A2AAgentFleet:
    """Start/stop a group of local A2A agents together."""

    def __init__(self, agents: list[A2AAgentProcess]):
        self.agents = agents

    def start_all(self, timeout: float = 20.0) -> None:
        for a in self.agents:
            a.start()
        for a in self.agents:
            if not a.wait_ready(timeout=timeout):
                self.stop_all()
                raise RuntimeError(
                    f"A2A agent on port {a.port} ({a.module}) failed to become ready"
                )

    def stop_all(self) -> None:
        for a in self.agents:
            a.stop()
