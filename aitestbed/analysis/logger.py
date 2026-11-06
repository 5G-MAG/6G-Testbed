"""
Traffic Logger for the 6G AI Traffic Testbed.

Logs all API interactions with timing metrics to SQLite database.
"""

import sqlite3
import time
import json
from dataclasses import dataclass, asdict, field
from typing import Optional, Any
from pathlib import Path
from contextlib import contextmanager


@dataclass
class LogRecord:
    """
    A single traffic log record capturing all relevant metrics.

    Aligned with 3GPP 6G Media Study metrics requirements.
    """
    # Identification
    timestamp: float
    scenario_id: str
    session_id: str
    turn_index: int
    run_index: int = 0

    # Provider info
    provider: str = ""
    model: str = ""

    # Traffic metrics
    request_bytes: int = 0
    response_bytes: int = 0

    # Token metrics
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None

    # Timing metrics (all in seconds)
    t_request_start: float = 0.0
    t_first_token: Optional[float] = None
    t_last_token: Optional[float] = None
    latency_sec: float = 0.0

    # Network context
    network_profile: str = ""

    # Response status
    http_status: int = 200
    error_type: Optional[str] = None
    success: bool = True

    # Agent/Tool metrics
    tool_calls_count: int = 0
    total_tool_bytes: int = 0
    tool_latency_sec: float = 0.0

    # Streaming metrics
    is_streaming: bool = False
    chunk_count: int = 0
    inter_chunk_times: str = ""  # JSON array of inter-chunk times

    # Additional metadata
    metadata: str = ""  # JSON for extra data

    @property
    def ttft(self) -> Optional[float]:
        """Time to first token in seconds."""
        if self.t_first_token is None:
            return None
        return self.t_first_token - self.t_request_start

    @property
    def ttlt(self) -> Optional[float]:
        """Time to last token in seconds."""
        if self.t_last_token is None:
            return None
        return self.t_last_token - self.t_request_start

    @property
    def total_bytes(self) -> int:
        """Total bytes (request + response)."""
        return self.request_bytes + self.response_bytes

    @property
    def ul_dl_ratio(self) -> float:
        """Uplink to downlink ratio."""
        if self.response_bytes == 0:
            return float('inf') if self.request_bytes > 0 else 0.0
        return self.request_bytes / self.response_bytes


class TrafficLogger:
    """
    SQLite-based traffic logger for the testbed.

    Provides thread-safe logging and export capabilities.
    """

    def __init__(self, db_path: str = "logs/traffic_logs.db"):
        """
        Initialize the logger.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
        with self._get_connection() as conn:
            conn.execute("""
            CREATE TABLE IF NOT EXISTS traffic_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                scenario_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                turn_index INTEGER NOT NULL,
                run_index INTEGER DEFAULT 0,

                provider TEXT,
                model TEXT,

                request_bytes INTEGER DEFAULT 0,
                response_bytes INTEGER DEFAULT 0,

                tokens_in INTEGER,
                tokens_out INTEGER,

                t_request_start REAL,
                t_first_token REAL,
                t_last_token REAL,
                latency_sec REAL,

                network_profile TEXT,

                http_status INTEGER DEFAULT 200,
                error_type TEXT,
                success INTEGER DEFAULT 1,

                tool_calls_count INTEGER DEFAULT 0,
                total_tool_bytes INTEGER DEFAULT 0,
                tool_latency_sec REAL DEFAULT 0,

                is_streaming INTEGER DEFAULT 0,
                chunk_count INTEGER DEFAULT 0,
                inter_chunk_times TEXT,

                metadata TEXT,

                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """)

            # Create indexes for common queries
            conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_scenario_profile
            ON traffic_logs(scenario_id, network_profile)
            """)

            conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_session
            ON traffic_logs(session_id)
            """)

            conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp
            ON traffic_logs(timestamp)
            """)

            conn.commit()

    @contextmanager
    def _get_connection(self):
        """Get a database connection with proper cleanup."""
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def log(self, record: LogRecord) -> int:
        """
        Log a single record.

        Args:
            record: LogRecord to store

        Returns:
            ID of the inserted record
        """
        with self._get_connection() as conn:
            cursor = conn.execute("""
            INSERT INTO traffic_logs (
                timestamp, scenario_id, session_id, turn_index, run_index,
                provider, model,
                request_bytes, response_bytes,
                tokens_in, tokens_out,
                t_request_start, t_first_token, t_last_token, latency_sec,
                network_profile,
                http_status, error_type, success,
                tool_calls_count, total_tool_bytes, tool_latency_sec,
                is_streaming, chunk_count, inter_chunk_times,
                metadata
            ) VALUES (
                :timestamp, :scenario_id, :session_id, :turn_index, :run_index,
                :provider, :model,
                :request_bytes, :response_bytes,
                :tokens_in, :tokens_out,
                :t_request_start, :t_first_token, :t_last_token, :latency_sec,
                :network_profile,
                :http_status, :error_type, :success,
                :tool_calls_count, :total_tool_bytes, :tool_latency_sec,
                :is_streaming, :chunk_count, :inter_chunk_times,
                :metadata
            )
            """, asdict(record))
            conn.commit()
            return cursor.lastrowid

    def log_batch(self, records: list[LogRecord]) -> None:
        """Log multiple records in a single transaction."""
        with self._get_connection() as conn:
            for record in records:
                conn.execute("""
                INSERT INTO traffic_logs (
                    timestamp, scenario_id, session_id, turn_index, run_index,
                    provider, model,
                    request_bytes, response_bytes,
                    tokens_in, tokens_out,
                    t_request_start, t_first_token, t_last_token, latency_sec,
                    network_profile,
                    http_status, error_type, success,
                    tool_calls_count, total_tool_bytes, tool_latency_sec,
                    is_streaming, chunk_count, inter_chunk_times,
                    metadata
                ) VALUES (
                    :timestamp, :scenario_id, :session_id, :turn_index, :run_index,
                    :provider, :model,
                    :request_bytes, :response_bytes,
                    :tokens_in, :tokens_out,
                    :t_request_start, :t_first_token, :t_last_token, :latency_sec,
                    :network_profile,
                    :http_status, :error_type, :success,
                    :tool_calls_count, :total_tool_bytes, :tool_latency_sec,
                    :is_streaming, :chunk_count, :inter_chunk_times,
                    :metadata
                )
                """, asdict(record))
            conn.commit()

    def query(
        self,
        scenario_id: Optional[str] = None,
        network_profile: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 1000
    ) -> list[dict]:
        """
        Query log records with optional filters.

        Returns:
            List of records as dictionaries
        """
        query = "SELECT * FROM traffic_logs WHERE 1=1"
        params = {}

        if scenario_id:
            query += " AND scenario_id = :scenario_id"
            params["scenario_id"] = scenario_id

        if network_profile:
            query += " AND network_profile = :network_profile"
            params["network_profile"] = network_profile

        if session_id:
            query += " AND session_id = :session_id"
            params["session_id"] = session_id

        query += f" ORDER BY timestamp DESC LIMIT {limit}"

        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def get_scenario_summary(self, scenario_id: str) -> dict:
        """Get summary statistics for a scenario."""
        with self._get_connection() as conn:
            cursor = conn.execute("""
            SELECT
                network_profile,
                COUNT(*) as count,
                AVG(latency_sec) as avg_latency,
                MIN(latency_sec) as min_latency,
                MAX(latency_sec) as max_latency,
                AVG(request_bytes) as avg_request_bytes,
                AVG(response_bytes) as avg_response_bytes,
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as success_rate,
                AVG(tokens_in) as avg_tokens_in,
                AVG(tokens_out) as avg_tokens_out
            FROM traffic_logs
            WHERE scenario_id = ?
            GROUP BY network_profile
            """, (scenario_id,))

            results = {}
            for row in cursor.fetchall():
                results[row["network_profile"]] = dict(row)

            return results

    def export_csv(self, output_path: str, scenario_id: Optional[str] = None) -> None:
        """Export logs to CSV file."""
        import csv

        records = self.query(scenario_id=scenario_id, limit=100000)

        if not records:
            return

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)

    def export_parquet(self, output_path: str, scenario_id: Optional[str] = None) -> None:
        """Export logs to Parquet file (requires pyarrow)."""
        import pandas as pd

        records = self.query(scenario_id=scenario_id, limit=100000)

        if not records:
            return

        df = pd.DataFrame(records)
        df.to_parquet(output_path, index=False)

    def clear(self, scenario_id: Optional[str] = None) -> int:
        """
        Clear logs, optionally for a specific scenario.

        Returns:
            Number of records deleted
        """
        with self._get_connection() as conn:
            if scenario_id:
                cursor = conn.execute(
                    "DELETE FROM traffic_logs WHERE scenario_id = ?",
                    (scenario_id,)
                )
            else:
                cursor = conn.execute("DELETE FROM traffic_logs")
            conn.commit()
            return cursor.rowcount
