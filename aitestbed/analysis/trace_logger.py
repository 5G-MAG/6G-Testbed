"""
Trace logger for capturing request/response payloads to disk.
"""

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


_SENSITIVE_KEY_PARTS = (
    "authorization", "api_key", "apikey", "access_token", "refresh_token",
    "password", "passwd", "secret", "cookie", "credential", "signature",
)
_BEARER_RE = re.compile(r"(?i)\b(bearer|basic)\s+[A-Za-z0-9._~+/=-]+")


def _redact(value: Any, key: str = "") -> Any:
    """Recursively redact credential-bearing fields before persistence."""
    lowered = key.lower().replace("-", "_")
    if any(part in lowered for part in _SENSITIVE_KEY_PARTS):
        return "[REDACTED]"
    if isinstance(value, dict):
        return {str(k): _redact(v, str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact(item) for item in value]
    if isinstance(value, str):
        return _BEARER_RE.sub(lambda match: f"{match.group(1)} [REDACTED]", value)
    return value


@dataclass
class TraceLogger:
    """Write detailed request/response traces to disk when enabled."""
    log_dir: Path
    enabled: bool = False
    max_events: Optional[int] = None
    max_bytes: Optional[int] = None

    @classmethod
    def from_env(cls) -> "TraceLogger":
        enabled = os.environ.get("TRACE_PAYLOADS", "0").lower() in ("1", "true", "yes")
        log_dir = Path(os.environ.get("TRACE_LOG_DIR", "logs/traces"))
        max_events = _parse_int_env("TRACE_MAX_EVENTS")
        max_bytes = _parse_int_env("TRACE_MAX_BYTES")
        return cls(log_dir=log_dir, enabled=enabled, max_events=max_events, max_bytes=max_bytes)

    def write_trace(
        self,
        *,
        scenario_id: str,
        session_id: str,
        turn_index: int,
        run_index: int,
        network_profile: str,
        provider: str,
        model: str,
        request_payload: Optional[Any] = None,
        response_payload: Optional[Any] = None,
        response_events: Optional[list[Any]] = None,
        note: Optional[str] = None,
        timing: Optional[dict[str, Any]] = None,
        metrics: Optional[dict[str, Any]] = None,
    ) -> Optional[str]:
        if not self.enabled:
            return None
        if request_payload is None and response_payload is None and not response_events:
            return None

        self.log_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.log_dir, 0o700)
        except OSError:
            pass

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        session_tag = session_id.replace("-", "")[:8]
        scenario_tag = (scenario_id or "scenario").replace(" ", "_")
        event_tag = f"{time.time_ns() % 1_000_000_000:09d}"
        file_name = (
            f"{timestamp}_{scenario_tag}_{session_tag}_"
            f"turn{turn_index:03d}_{event_tag}.json"
        )
        path = self.log_dir / file_name

        events = list(response_events or [])
        if self.max_events and len(events) > self.max_events:
            events = events[: self.max_events]

        trace = {
            "trace_schema_version": "1.0",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "scenario_id": scenario_id,
            "session_id": session_id,
            "turn_index": turn_index,
            "run_index": run_index,
            "network_profile": network_profile,
            "provider": provider,
            "model": model,
            "note": note,
            "request": _redact(request_payload),
            "response": _redact(response_payload),
            "response_events": _redact(events),
            "timing": timing or {},
            "metrics": metrics or {},
        }

        payload = json.dumps(trace, indent=2, ensure_ascii=True, default=str)
        if self.max_bytes and len(payload.encode("utf-8")) > self.max_bytes:
            trace["response_events"] = []
            trace["note"] = (note or "") + " [trace truncated: TRACE_MAX_BYTES]"
            payload = json.dumps(trace, indent=2, ensure_ascii=True, default=str)

        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
        return str(path)


def _parse_int_env(key: str) -> Optional[int]:
    raw = os.environ.get(key, "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value or None
