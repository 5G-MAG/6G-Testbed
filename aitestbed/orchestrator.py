#!/usr/bin/env python3
"""
Main Orchestrator for the 6G AI Traffic Testbed.

Coordinates scenario execution, network emulation, and data collection.
"""

import argparse
import logging
import time
import yaml
import json
import signal
from pathlib import Path
from typing import Optional
from datetime import datetime
from dataclasses import asdict

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()  # Loads from .env in current directory
load_dotenv(Path(__file__).parent / ".env")  # Also try aitestbed/.env
load_dotenv(Path(__file__).parent.parent / ".env")  # Also try repo root .env

from clients import OpenAIClient, GeminiClient, DeepSeekClient, VLLMClient, AzureOpenAIClient, AzureInferenceClient, AnthropicClient
from analysis import TrafficLogger, MetricsCalculator, LogRecord
from configs import DEFAULT_CAPTURE_FILTER
from netemu import NetworkEmulator
from scenarios import (
    ChatScenario,
    ShoppingAgentScenario,
    WebSearchAgentScenario,
    GeneralAgentScenario,
    ImageGenerationScenario,
    MultimodalScenario,
    VideoUnderstandingScenario,
    ScenarioResult,
    DirectWebSearchScenario,
    ParallelSearchBenchmarkScenario,
    RealtimeConversationScenario,
    RealtimeWebRTCConversationScenario,
    RealtimeAudioScenario,
    RealtimeAudioWebRTCScenario,
    ComputerUseScenario,
    MusicAgentScenario,
    MusicResearchAgentScenario,
    PlaywrightAgentScenario,
    TradingAgentScenario,
    WeatherAgentScenario,
    NavigationWeatherAgentScenario,
    MapsAgentScenario,
    ExaSearchAgentScenario,
    TwilioCommunicationAgentScenario,
    SmartHomeAgentScenario,
    OpenClawScenario,
    A2ASingleTaskScenario,
    A2AStreamingScenario,
    A2AMultiAgentScenario)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("orchestrator")
APP_DIR = Path(__file__).resolve().parent


class RunFailedError(Exception):
    """Raised when a run fails and --stop-on-error is active."""
    pass


class TestbedOrchestrator:
    """
    Main orchestrator for running traffic characterization experiments.
    """

    def __init__(
        self,
        config_path: str = "configs/scenarios.yaml",
        profiles_path: str = "configs/profiles.yaml",
        db_path: str = "logs/traffic_logs.db",
        network_interface: str = "auto",
        egress_only: bool = False,
        mcp_transport: str = "http"):
        """
        Initialize the orchestrator.

        Args:
            config_path: Path to scenarios configuration
            profiles_path: Path to network profiles configuration
            db_path: Path to SQLite database for logs
            network_interface: Network interface for emulation
            egress_only: If True, only shape egress traffic (default: False, shapes both)
        """
        self.config_path = Path(config_path)
        self.profiles_path = Path(profiles_path)

        # Load configurations
        with open(self.config_path) as f:
            self.scenarios_config = yaml.safe_load(f)
        # Re-parse profiles.yaml as a raw dict so we retain access to optional
        # `uplink:` blocks (netemu's NetworkProfile dataclass strips them).
        with open(self.profiles_path) as f:
            self.profiles_config = yaml.safe_load(f) or {}

        # Initialize components
        self.logger = TrafficLogger(db_path)
        self.emulator = NetworkEmulator(
            interface=network_interface,
            profiles_path=str(self.profiles_path),
            bidirectional=not egress_only
        )
        # Expose the resolved interface (in case "auto" was passed)
        self.network_interface = self.emulator.interface
        self.mcp_transport = mcp_transport

        # Validate any asymmetric `uplink:` blocks now so bad YAML fails fast.
        self._validate_asymmetric_profiles()

        # Initialize clients (lazy loaded based on provider)
        self._clients = {}

        # Scenario type mapping
        self.scenario_classes = {
            "chat": ChatScenario,
            "agent": ShoppingAgentScenario,
            "shopping_agent": ShoppingAgentScenario,
            "web_search_agent": WebSearchAgentScenario,
            "general_agent": GeneralAgentScenario,
            "image": ImageGenerationScenario,
            "image_generation": ImageGenerationScenario,
            "multimodal": MultimodalScenario,
            "video_understanding": VideoUnderstandingScenario,
            "video": VideoUnderstandingScenario,
            "computer_use": ComputerUseScenario,
            # Music agent scenarios (Spotify MCP)
            "music_agent": MusicAgentScenario,
            "music_research_agent": MusicResearchAgentScenario,
            # Playwright browser automation agent
            "playwright_agent": PlaywrightAgentScenario,
            # Trading / market data agent (Alpaca MCP)
            "trading_agent": TradingAgentScenario,
            # Weather / environment agent (Open-Meteo, no API key)
            "weather_agent": WeatherAgentScenario,
            # Combined navigation + weather agent (Google Maps + Open-Meteo)
            "navigation_weather_agent": NavigationWeatherAgentScenario,
            # Maps / navigation agent (Google Maps)
            "maps_agent": MapsAgentScenario,
            # Exa.ai search agent
            "exa_search_agent": ExaSearchAgentScenario,
            # Twilio communications agent (SMS / WhatsApp)
            "twilio_agent": TwilioCommunicationAgentScenario,
            # Smart home agent (Home Assistant)
            "smart_home_agent": SmartHomeAgentScenario,
            # OpenClaw local personal-assistant agent (gateway on lo:18789)
            "openclaw_agent": OpenClawScenario,
            # A2A (Agent2Agent protocol) scenarios
            "a2a_single_task": A2ASingleTaskScenario,
            "a2a_streaming": A2AStreamingScenario,
            "a2a_multi_agent": A2AMultiAgentScenario,
            # Direct search scenarios (no MCP)
            "direct_search": DirectWebSearchScenario,
            "direct_web_search": DirectWebSearchScenario,
            "parallel_search_benchmark": ParallelSearchBenchmarkScenario,
            # Real-time conversational AI scenarios
            "realtime_conversation": RealtimeConversationScenario,
            "realtime_conversation_webrtc": RealtimeWebRTCConversationScenario,
            "realtime_audio": RealtimeAudioScenario,
            "realtime_audio_webrtc": RealtimeAudioWebRTCScenario,
        }

    # ------------------------------------------------------------------
    # Asymmetric profile support (S4-260848 Table C.Z-1)
    # ------------------------------------------------------------------

    # Fields that may legally appear at the top level of a profile or
    # inside an `uplink:` block. Mirrors NetworkProfile dataclass + the
    # kwargs accepted by NetworkEmulator.apply_settings(). `description`
    # and `uplink` are intentionally excluded.
    _PROFILE_IMPAIRMENT_FIELDS = frozenset({
        "delay_ms", "jitter_ms",
        "delay_distribution", "delay_correlation_pct",
        "loss_pct", "loss_correlation_pct", "loss_model",
        "rate_mbit", "rate_ceil_mbit",
        "rate_burst_kbit", "rate_cburst_kbit",
        "corruption_pct", "corruption_correlation_pct",
        "reorder_pct", "reorder_correlation_pct",
        "duplicate_pct", "duplicate_correlation_pct",
        "limit_packets",
    })

    def _validate_asymmetric_profiles(self) -> None:
        """Reject malformed `uplink:` blocks at startup.

        Catches: non-dict uplink, nested uplink, unknown fields. Empty
        uplink (`uplink: {}`) is allowed and logged, treated as symmetric.
        """
        profiles = (self.profiles_config or {}).get("profiles", {}) or {}
        for name, raw in profiles.items():
            if not isinstance(raw, dict):
                continue
            ul = raw.get("uplink")
            if ul is None:
                continue
            if not isinstance(ul, dict):
                raise ValueError(
                    f"profile '{name}': 'uplink:' must be a mapping, "
                    f"got {type(ul).__name__}"
                )
            if not ul:
                logger.warning(
                    f"profile '{name}': empty uplink: block, treating as symmetric"
                )
                continue
            if "uplink" in ul:
                raise ValueError(
                    f"profile '{name}': nested 'uplink:' is forbidden"
                )
            unknown = set(ul) - self._PROFILE_IMPAIRMENT_FIELDS
            if unknown:
                raise ValueError(
                    f"profile '{name}': unknown key(s) under uplink: "
                    f"{sorted(unknown)}"
                )

    def _apply_emulator_profile(
        self, profile_name: str, ingress_profile: Optional[str]
    ) -> bool:
        """Apply *profile_name*, honoring the optional `uplink:` override.

        Symmetric profiles fall through to ``emulator.apply_profile`` (existing
        behavior). Asymmetric profiles are applied via ``apply_settings``,
        which already supports an ``ingress_settings`` dict, no synthetic
        netemu profiles are registered.

        CLI ``--ingress-profile Y`` wins over a built-in ``uplink:`` block:
        egress still uses (base ⊕ uplink), ingress uses Y's base fields.

        ``no_emulation`` is a true bypass: any prior tc state is cleared and
        no qdisc is added, so the device is observed exactly as the kernel
        sees it (no netem queueing, no IFB).
        """
        # No-emulation reference baseline: skip tc/netem entirely.
        if profile_name == "no_emulation":
            logger.info(
                "Profile no_emulation: clearing prior tc state and bypassing netem"
            )
            self.emulator.clear()
            return True

        profiles = (self.profiles_config or {}).get("profiles", {}) or {}
        raw = profiles.get(profile_name) if isinstance(profiles, dict) else None
        uplink = raw.get("uplink") if isinstance(raw, dict) else None

        # Symmetric fast path
        if not uplink:
            return self.emulator.apply_profile(
                profile_name, ingress_profile=ingress_profile
            )

        # Asymmetric, split base + uplink overrides into egress/ingress dicts
        base = {
            k: v for k, v in raw.items()
            if k in self._PROFILE_IMPAIRMENT_FIELDS
        }
        egress = {**base, **uplink}

        if ingress_profile == "none":
            ingress_dict = None
            disable_ingress = True
        elif ingress_profile:
            cli_raw = profiles.get(ingress_profile)
            if not isinstance(cli_raw, dict):
                raise ValueError(
                    f"--ingress-profile '{ingress_profile}' not found in profiles.yaml"
                )
            ingress_dict = {
                k: v for k, v in cli_raw.items()
                if k in self._PROFILE_IMPAIRMENT_FIELDS
            }
            disable_ingress = False
        else:
            # Default: ingress uses the base (downlink) fields unchanged
            ingress_dict = base
            disable_ingress = False

        logger.info(
            f"Profile {profile_name} is asymmetric. "
            f"uplink overrides: {sorted(uplink.keys())}"
        )
        if not self.emulator.bidirectional:
            logger.warning(
                f"Profile {profile_name} has an uplink: block but --egress-only "
                f"is set, downlink (base) side will not be applied."
            )
        logger.debug(f"  egress (UL):  {egress}")
        logger.debug(
            f"  ingress (DL): "
            f"{ingress_dict if ingress_dict is not None else '<egress-only>'}"
        )

        return self.emulator.apply_settings(
            **egress,
            profile_name=profile_name,
            ingress_settings=ingress_dict,
            disable_ingress=disable_ingress)

    def get_client(self, provider: str):
        """Get or create an LLM client for the given provider."""
        if provider not in self._clients:
            if provider == "openai":
                self._clients[provider] = OpenAIClient()
            elif provider == "gemini":
                self._clients[provider] = GeminiClient()
            elif provider == "deepseek":
                self._clients[provider] = DeepSeekClient()
            elif provider == "vllm":
                self._clients[provider] = VLLMClient()
            elif provider == "azure_openai":
                self._clients[provider] = AzureOpenAIClient()
            elif provider == "azure_inference":
                self._clients[provider] = AzureInferenceClient()
            elif provider == "anthropic":
                self._clients[provider] = AnthropicClient()
            elif provider in ("none", "null", "a2a", "openclaw"):
                # Scenarios that do not call an LLM through the testbed layer
                # (A2A agent-to-agent, OpenClaw's own model loop). Use a stub
                # so construction never requires an API key.
                from clients.base import NullLLMClient
                self._clients[provider] = NullLLMClient(provider)
            else:
                raise ValueError(f"Unknown provider: {provider}")
        return self._clients[provider]

    def scenarios_use_loopback(self, scenario_arg: Optional[str]) -> bool:
        """True if the selected scenario(s) declare ``loopback_ports`` (or
        ``uses_loopback``) in their config.

        Such scenarios (OpenClaw, local A2A agents) talk to a server bound on
        the loopback interface that must be captured and shaped exactly like
        MCP-over-HTTP. For the test matrix (``--scenario all``) this returns
        True if *any* matrix scenario needs loopback handling.
        """
        scenarios = self.scenarios_config.get("scenarios", {}) or {}

        def _needs(name: Optional[str]) -> bool:
            cfg = scenarios.get(name) or {}
            return bool(cfg.get("loopback_ports")) or bool(cfg.get("uses_loopback"))

        if scenario_arg == "all":
            matrix = self.scenarios_config.get("test_matrix", []) or []
            return any(_needs(e.get("scenario")) for e in matrix)
        if scenario_arg:
            return _needs(scenario_arg)
        return False

    def get_completed_runs(self, scenario_name: str, profile_name: str) -> int:
        """Query the DB for the number of completed runs (successful or timed out).

        A "completed run" is a distinct session_id for the given
        scenario+profile that either succeeded on every record OR is a
        timeout placeholder.  Timed-out runs are legitimate data points
        (the network conditions *are* the cause) and must not be retried
        on resume.
        """
        import sqlite3
        try:
            db_path = self.logger.db_path
            with sqlite3.connect(db_path) as conn:
                cursor = conn.execute("""
                    SELECT COUNT(DISTINCT session_id) FROM traffic_logs
                    WHERE scenario_id = ?
                      AND network_profile = ?
                      AND session_id NOT LIKE 'pcap_%'
                      AND (
                          session_id LIKE 'timeout_%'
                          OR session_id NOT IN (
                              SELECT DISTINCT session_id FROM traffic_logs
                              WHERE scenario_id = ?
                                AND network_profile = ?
                                AND success = 0
                          )
                      )
                """, (scenario_name, profile_name, scenario_name, profile_name))
                count = cursor.fetchone()[0]
                return count
        except Exception:
            return 0

    def run_experiment(
        self,
        scenario_name: str,
        profile_name: str,
        runs: int = 10,
        inter_run_delay: float = 1.0,
        ingress_profile: Optional[str] = None,
        run_timeout: Optional[float] = None,
        stop_on_error: bool = False,
        resume: bool = False) -> list[ScenarioResult]:
        """
        Run a single experiment (scenario + profile combination).

        Args:
            scenario_name: Name of the scenario to run
            profile_name: Network profile to apply (egress, and ingress if not specified separately)
            runs: Number of runs
            inter_run_delay: Delay between runs in seconds
            ingress_profile: Optional separate profile for ingress traffic.
                            None = use same as egress, "none" = no ingress shaping
            run_timeout: Per-run timeout in seconds. None means no timeout.
            stop_on_error: If True, raise RunFailedError on the first failed run
                          (after retries are exhausted).
            resume: If True, query the DB for already-completed successful runs
                   and skip them.

        Returns:
            List of ScenarioResult objects

        Raises:
            RunFailedError: If stop_on_error is True and a run fails.
        """
        # Get scenario configuration
        scenario_config = self.scenarios_config["scenarios"].get(scenario_name)
        if not scenario_config:
            raise ValueError(f"Unknown scenario: {scenario_name}")

        if scenario_config.get("disabled", False):
            logger.info(f"Skipping disabled scenario: {scenario_name}")
            return []

        # Determine scenario type and class
        scenario_type = scenario_config.get("type", "chat")
        scenario_class = self.scenario_classes.get(scenario_type)
        if not scenario_class:
            raise ValueError(f"Unknown scenario type: {scenario_type}")

        defaults = self.scenarios_config.get("defaults", {})
        retry_count = int(defaults.get("retry_count", 0))
        retry_backoff_sec = float(defaults.get("retry_backoff_sec", 1.0))
        retry_backoff_multiplier = float(defaults.get("retry_backoff_multiplier", 2.0))

        # Resume: check before any expensive setup (client init, network
        # profile application) so fully-completed combos are skipped cheaply.
        start_run = 0
        if resume:
            completed = self.get_completed_runs(scenario_name, profile_name)
            if completed >= runs:
                logger.info(
                    f"Skipping {scenario_name}/{profile_name}: "
                    f"all {runs} runs already completed ({completed} in DB)"
                )
                return []
            if completed > 0:
                start_run = completed
                logger.info(
                    f"Resuming {scenario_name}/{profile_name} from run "
                    f"{start_run + 1}/{runs} ({completed} already completed)"
                )

        # Get provider and client
        provider = scenario_config.get("provider", "openai")
        client = self.get_client(provider)

        # Create scenario instance
        scenario_config["scenario_id"] = scenario_name
        # Inject MCP transport setting for agent scenarios
        if self.mcp_transport != "stdio" and "mcp_transport" not in scenario_config:
            scenario_config["mcp_transport"] = self.mcp_transport
        scenario = scenario_class(client, self.logger, scenario_config)

        # Pass emulator to agent scenarios so they can apply netem to loopback
        if hasattr(scenario, "emulator"):
            scenario.emulator = self.emulator
            scenario._current_network_profile = profile_name

        # Apply network profile
        if ingress_profile:
            logger.info(f"Applying network profile: egress={profile_name}, ingress={ingress_profile}")
        else:
            logger.info(f"Applying network profile: {profile_name} (bidirectional)")
        if not self._apply_emulator_profile(profile_name, ingress_profile):
            logger.warning(f"Failed to apply network profile (may require sudo)")

        results = []

        try:
            for run_index in range(start_run, runs):
                logger.info(f"Running {scenario_name} [{run_index + 1}/{runs}] with profile {profile_name}")

                attempt = 0
                retry_reason = None
                while True:
                    t_start = time.time()

                    if run_timeout is not None:
                        result = self._run_with_timeout(
                            scenario, profile_name, run_index, run_timeout,
                            scenario_name)
                    else:
                        result = scenario.run(network_profile=profile_name, run_index=run_index)

                    t_elapsed = time.time() - t_start

                    current_reason = self._get_retry_reason(result)
                    if current_reason:
                        retry_reason = current_reason

                    if result.success or attempt >= retry_count or not current_reason:
                        break

                    backoff = retry_backoff_sec * (retry_backoff_multiplier ** attempt)
                    logger.warning(
                        f"  Retry {attempt + 1}/{retry_count} after {backoff:.1f}s "
                        f"({current_reason})"
                    )
                    time.sleep(backoff)
                    attempt += 1

                results.append(result)

                # Persist timeout results to the DB so that --resume
                # counts them as completed and does not retry them.
                if not result.log_records and result.session_id.startswith("timeout_"):
                    timeout_record = LogRecord(
                        timestamp=time.time(),
                        scenario_id=scenario_name,
                        session_id=result.session_id,
                        turn_index=0,
                        run_index=run_index,
                        provider=scenario_config.get("provider", ""),
                        model=scenario_config.get("model", ""),
                        network_profile=profile_name,
                        latency_sec=result.total_latency_sec,
                        success=False,
                        error_type="timeout")
                    self.logger.log(timeout_record)

                logger.info(
                    f"  Completed: success={result.success}, "
                    f"latency={result.total_latency_sec:.2f}s, "
                    f"api_calls={result.api_call_count}, "
                    f"tools={result.tool_calls_count}"
                )
                if not result.success and result.error_message:
                    logger.error(f"  Error: {result.error_message}")
                if attempt > 0:
                    result.metadata["retry_attempts"] = attempt
                    if retry_reason:
                        result.metadata["retry_reason"] = retry_reason

                if run_index < runs - 1:
                    time.sleep(inter_run_delay)

        finally:
            # Clear network profile
            self.emulator.clear()

        # stop-on-error: only raise if every run failed on a baseline profile
        # (no_emulation), indicating a real infrastructure problem. Failures on
        # degraded profiles (satellite, congested, cell_edge, etc.) are expected
        # data points, the network conditions *are* the cause.
        baseline_profiles = {"no_emulation", "ideal_6g"}
        if (
            stop_on_error
            and results
            and not any(r.success for r in results)
            and profile_name in baseline_profiles
        ):
            raise RunFailedError(
                f"{scenario_name}/{profile_name}: all {len(results)} runs failed. "
                f"Last error: {results[-1].error_message}"
            )

        return results

    def _run_with_timeout(
        self,
        scenario,
        profile_name: str,
        run_index: int,
        timeout_sec: float,
        scenario_name: str) -> ScenarioResult:
        """Run one scenario in a killable process group.

        A timed-out thread cannot be cancelled and used to continue issuing API
        calls and DB writes during later runs. On POSIX the forked worker starts
        a new process group; timeout cleanup terminates that whole group,
        including MCP/browser children spawned by the scenario.
        """
        import multiprocessing
        import os
        import traceback

        if os.name != "posix":
            raise RuntimeError("Per-run timeout isolation currently requires POSIX")
        ctx = multiprocessing.get_context("fork")
        recv_conn, send_conn = ctx.Pipe(duplex=False)

        def _run_in_process():
            import asyncio
            try:
                os.setsid()
            except OSError:
                pass
            try:
                asyncio.get_event_loop()
            except RuntimeError:
                asyncio.set_event_loop(asyncio.new_event_loop())
            try:
                r = scenario.run(
                    network_profile=profile_name, run_index=run_index
                )
                send_conn.send(("ok", r))
            except BaseException as exc:  # noqa: BLE001, forward everything
                send_conn.send((
                    "err",
                    {
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                ))
            finally:
                send_conn.close()

        worker = ctx.Process(
            target=_run_in_process,
            name=f"scenario-{scenario_name}-run{run_index}",
            daemon=False,
        )
        worker.start()
        send_conn.close()

        if not recv_conn.poll(timeout_sec):
            logger.warning(
                f"  Run {run_index + 1} of {scenario_name} timed out "
                f"after {timeout_sec:.0f}s; terminating worker process group"
            )
            try:
                if os.getpgid(worker.pid) == worker.pid:
                    os.killpg(worker.pid, signal.SIGTERM)
                else:
                    worker.terminate()
            except (ProcessLookupError, OSError):
                worker.terminate()
            worker.join(timeout=5)
            if worker.is_alive():
                try:
                    if os.getpgid(worker.pid) == worker.pid:
                        os.killpg(worker.pid, signal.SIGKILL)
                    else:
                        worker.kill()
                except (ProcessLookupError, OSError):
                    worker.kill()
                worker.join(timeout=5)
            recv_conn.close()
            return ScenarioResult(
                scenario_id=scenario_name,
                session_id=f"timeout_{int(time.time())}",
                network_profile=profile_name,
                run_index=run_index,
                success=False,
                total_latency_sec=timeout_sec,
                error_message=f"Run timed out after {timeout_sec:.0f}s")

        try:
            kind, payload = recv_conn.recv()
        except EOFError as exc:
            worker.join(timeout=5)
            raise RuntimeError(
                f"Scenario worker exited with code {worker.exitcode} without a result"
            ) from exc
        finally:
            recv_conn.close()
        worker.join(timeout=5)
        if kind == "err":
            raise RuntimeError(
                f"Scenario worker failed ({payload['type']}): {payload['message']}\n"
                f"{payload['traceback']}"
            )
        return payload  # type: ignore[return-value]

    def _get_retry_reason(self, result: ScenarioResult) -> Optional[str]:
        """Return a retry reason for transient failures.

        Run-level timeouts are not retried: on degraded profiles they are a
        legitimate outcome. Per-request HTTP timeouts surfaced as LogRecord
        error_type are also terminal here.
        """
        if not result.log_records:
            error_text = (result.error_message or "").lower()
            if "rate limit" in error_text or "429" in error_text:
                return "rate_limited"
            return None

        for record in result.log_records:
            if record.success:
                continue
            if self._is_tool_record(record):
                continue

            if record.http_status == 429:
                return "rate_limited"
            if record.http_status and 500 <= record.http_status < 600 and record.http_status != 501:
                return "server_error"

        return None

    def _is_tool_record(self, record: LogRecord) -> bool:
        """Detect MCP tool call records via metadata."""
        if not record.metadata:
            return False
        try:
            metadata = json.loads(record.metadata)
        except json.JSONDecodeError:
            return False
        return metadata.get("type") == "mcp_tool_call"

    def run_test_matrix(
        self,
        matrix: Optional[list[dict]] = None,
        runs_per_experiment: int = 10,
        run_timeout: Optional[float] = None,
        stop_on_error: bool = False,
        resume: bool = False) -> dict:
        """
        Run a full test matrix.

        Args:
            matrix: Test matrix definition (uses config if not provided)
            runs_per_experiment: Default runs per experiment
            run_timeout: Per-run timeout in seconds (None = no timeout)
            stop_on_error: If True, abort the entire matrix on the first failed run.
            resume: If True, skip scenario/profile combos already completed in the DB.

        Returns:
            Dictionary with all results and computed metrics
        """
        if matrix is None:
            matrix = self.scenarios_config.get("test_matrix", [])

        all_results = {}
        all_metrics = []

        start_time = datetime.now()
        logger.info(f"Starting test matrix with {len(matrix)} experiments")

        for entry in matrix:
            scenario_name = entry["scenario"]
            scenario_def = self.scenarios_config["scenarios"].get(scenario_name, {})
            if scenario_def.get("disabled", False):
                logger.info(f"Skipping disabled scenario: {scenario_name}")
                continue
            profiles = entry.get("profiles", ["ideal_6g"])
            runs = entry.get("runs", runs_per_experiment)

            for profile_name in profiles:
                experiment_key = f"{scenario_name}_{profile_name}"
                logger.info(f"\n{'='*60}")
                logger.info(f"Experiment: {experiment_key}")
                logger.info(f"{'='*60}")

                try:
                    results = self.run_experiment(
                        scenario_name=scenario_name,
                        profile_name=profile_name,
                        runs=runs,
                        run_timeout=run_timeout,
                        stop_on_error=stop_on_error,
                        resume=resume)

                    all_results[experiment_key] = results

                    # Compute metrics
                    records = [
                        asdict(record)
                        for result in results
                        for record in result.log_records
                    ]
                    if not records:
                        records = [
                            {
                                "latency_sec": r.total_latency_sec,
                                "request_bytes": r.total_request_bytes,
                                "response_bytes": r.total_response_bytes,
                                "tokens_in": r.total_tokens_in,
                                "tokens_out": r.total_tokens_out,
                                "success": r.success,
                                "tool_calls_count": r.tool_calls_count,
                                "tool_latency_sec": (
                                    r.tool_total_latency_sec / r.tool_calls_count
                                    if r.tool_calls_count else 0.0
                                ),
                                "session_id": r.session_id,
                            }
                            for r in results
                        ]

                    metrics_defaults = self.scenarios_config.get("defaults", {})
                    metrics = MetricsCalculator.calculate(
                        records,
                        scenario_name,
                        profile_name,
                        stall_gap_sec=metrics_defaults.get("stall_gap_sec"),
                        burst_gap_sec=metrics_defaults.get("burst_gap_sec"))
                    all_metrics.append(metrics)

                    logger.info(f"Metrics: latency_mean={metrics.latency_mean:.3f}s, "
                               f"success_rate={metrics.success_rate:.1f}%")

                except RunFailedError:
                    raise
                except Exception as e:
                    logger.error(f"Experiment failed: {e}")
                    all_results[experiment_key] = {"error": str(e)}

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        logger.info(f"\n{'='*60}")
        logger.info(f"Test matrix completed in {duration:.1f}s")
        logger.info(f"{'='*60}")

        return {
            "results": all_results,
            "metrics": all_metrics,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_sec": duration
        }

    def generate_report(
        self,
        metrics: list,
        output_path: str = "results/reports/experiment_report.json"
    ) -> None:
        """Generate a JSON report from metrics."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        report = {
            "generated_at": datetime.now().isoformat(),
            "scenarios": {}
        }

        for m in metrics:
            if m.scenario_id not in report["scenarios"]:
                report["scenarios"][m.scenario_id] = {}

            report["scenarios"][m.scenario_id][m.network_profile] = \
                MetricsCalculator.to_3gpp_format(m)

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Report saved to {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="6G AI Traffic Characterization Testbed"
    )
    parser.add_argument(
        "--scenario", "-s",
        help="Scenario to run (or 'all' for test matrix)"
    )
    parser.add_argument(
        "--profile", "-p",
        default="no_emulation",
        help="Network profile to use"
    )
    parser.add_argument(
        "--runs", "-r",
        type=int,
        default=10,
        help="Number of runs per experiment"
    )
    parser.add_argument(
        "--config",
        default=str(APP_DIR / "configs" / "scenarios.yaml"),
        help="Path to scenarios config"
    )
    parser.add_argument(
        "--profiles",
        default=str(APP_DIR / "configs" / "profiles.yaml"),
        help="Path to network profiles config"
    )
    parser.add_argument(
        "--db",
        default="logs/traffic_logs.db",
        help="Path to SQLite database"
    )
    parser.add_argument(
        "--interface",
        default="auto",
        help="Network interface for emulation (auto = detect from default route)"
    )
    parser.add_argument(
        "--egress-only",
        action="store_true",
        help="Disable ingress shaping (only shape egress/outbound traffic)"
    )
    parser.add_argument(
        "--ingress-profile",
        default=None,
        help="Use a different network profile for ingress traffic (default: same as --profile)"
    )
    parser.add_argument(
        "--report",
        default="results/reports/experiment_report.json",
        help="Output path for report"
    )
    parser.add_argument(
        "--list-scenarios",
        action="store_true",
        help="List available scenarios"
    )
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="List available network profiles"
    )
    parser.add_argument(
        "--capture-pcap",
        action="store_true",
        help="Enable L3/L4 capture using tcpdump"
    )
    parser.add_argument(
        "--capture-filter",
        default=DEFAULT_CAPTURE_FILTER,
        help="tcpdump filter expression. The default captures HTTP(S) plus "
             "WebRTC ICE/SRTP/RTP UDP traffic. Pass an empty string to capture "
             "all traffic."
    )
    parser.add_argument(
        "--capture-dir",
        default="results/captures",
        help="Directory for pcap captures"
    )
    parser.add_argument(
        "--capture-loopback",
        action="store_true",
        default=True,
        help="When MCP transport is http, also capture loopback (lo) so MCP "
             "JSON-RPC frames are visible in pcap. Enabled by default. "
             "Use --no-capture-loopback to disable."
    )
    parser.add_argument(
        "--no-capture-loopback",
        dest="capture_loopback",
        action="store_false",
        help="Disable the secondary loopback capture (mirror flag)."
    )
    parser.add_argument(
        "--capture-loopback-filter",
        default=None,
        help="BPF filter for the loopback capture. Defaults to "
             "'not port 22 and not port 53'."
    )
    parser.add_argument(
        "--capture-l7",
        action="store_true",
        help="Enable L7 capture using mitmproxy"
    )
    parser.add_argument(
        "--capture-l7-dir",
        default="results/l7_captures",
        help="Directory for L7 captures"
    )
    parser.add_argument(
        "--capture-l7-hosts",
        default=None,
        help="Comma-separated hostnames to filter L7 capture"
    )
    parser.add_argument(
        "--capture-l7-proxy-port",
        type=int,
        default=8080,
        help="Port for mitmproxy"
    )
    parser.add_argument(
        "--capture-l7-web-port",
        type=int,
        default=8081,
        help="Port for mitmproxy web UI (0 to disable)"
    )
    parser.add_argument(
        "--mcp-transport",
        choices=["stdio", "http"],
        default="http",
        help="MCP server transport: stdio (pipe, no netem) or http (TCP, netem-shaped)"
    )
    parser.add_argument(
        "--run-timeout",
        type=float,
        default=None,
        help="Per-run timeout in seconds (default: no timeout). "
             "Individual runs that exceed this are marked as failed and skipped."
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately on the first failed run (after retries)."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from where a previous run left off. Skips scenario/profile "
             "combinations that already have enough successful runs in the database."
    )

    args = parser.parse_args()

    orchestrator = TestbedOrchestrator(
        config_path=args.config,
        profiles_path=args.profiles,
        db_path=args.db,
        network_interface=args.interface,
        egress_only=args.egress_only,
        mcp_transport=args.mcp_transport)

    if args.list_scenarios:
        print("\nAvailable scenarios:")
        for name, config in orchestrator.scenarios_config["scenarios"].items():
            disabled = " [DISABLED]" if config.get("disabled", False) else ""
            print(f"  - {name}: {config.get('description', 'No description')}{disabled}")
        return

    if args.list_profiles:
        print("\nAvailable network profiles:")
        for name in orchestrator.emulator.list_profiles():
            profile = orchestrator.emulator.get_profile(name)
            print(f"  - {name}: {profile.description}")
        return

    pcap_controller = None
    pcap_start_time = None
    pcap_file = None
    l7_controller = None
    pcap_lo_controller = None
    pcap_lo_file = None
    # run_id ties primary + loopback pcaps to the same orchestrator invocation,
    # so post-hoc joins do not have to rely on overlapping timestamps.
    run_id = f"run_{int(time.time())}_{abs(hash(repr(vars(args)))) % 100000:05d}"
    try:
        if args.capture_pcap:
            capture_meta = {
                "run_id": run_id,
                "scenario_id": args.scenario or "unknown",
                "network_profile": (
                    args.profile if args.scenario and args.scenario != "all"
                    else "multiple"
                ),
            }
            try:
                from capture import CaptureController
                pcap_controller = CaptureController(
                    interface=orchestrator.network_interface,
                    capture_dir=args.capture_dir
                )
                pcap_start_time = time.time()
                pcap_file = pcap_controller.start(
                    filter_expr=args.capture_filter, metadata=capture_meta
                )
                if not pcap_file:
                    logger.warning("Failed to start tcpdump capture")
                    pcap_start_time = None
            except Exception as e:
                logger.warning(f"Failed to start tcpdump capture: {e}")
                pcap_controller = None
                pcap_start_time = None

            # Secondary loopback capture for MCP-over-HTTP traffic. The MCP
            # client (clients/mcp_client.py:MCPHttpConnection) POSTs JSON-RPC
            # to 127.0.0.1:<auto-port>; that traffic only appears on `lo` and
            # is invisible to the primary capture when the primary interface
            # is anything other than `lo` itself.
            primary_iface = orchestrator.network_interface or ""
            # Loopback traffic appears for MCP-over-HTTP and for any scenario
            # that drives a local server on lo (OpenClaw gateway, local A2A
            # agents), those declare loopback_ports/uses_loopback in config.
            needs_loopback = (
                orchestrator.mcp_transport == "http"
                or orchestrator.scenarios_use_loopback(args.scenario)
            )
            if (
                args.capture_pcap
                and args.capture_loopback
                and primary_iface != "lo"
                and needs_loopback
            ):
                try:
                    from capture import CaptureController
                    pcap_lo_controller = CaptureController(
                        interface="lo",
                        capture_dir=args.capture_dir)
                    # Filter out SSH/DNS noise; keep everything else on lo
                    # (MCP HTTP servers use ephemeral ports we cannot enumerate
                    # ahead of time, so a port-range filter is not possible).
                    lo_filter = args.capture_loopback_filter or (
                        "not port 22 and not port 53"
                    )
                    lo_filename = (
                        f"capture_lo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pcap"
                    )
                    pcap_lo_file = pcap_lo_controller.start(
                        filename=lo_filename,
                        filter_expr=lo_filter,
                        metadata={**capture_meta, "interface": "lo"},
                    )
                    if not pcap_lo_file:
                        logger.warning("Failed to start loopback tcpdump capture")
                        pcap_lo_controller = None
                except Exception as e:
                    logger.warning(f"Failed to start loopback tcpdump capture: {e}")
                    pcap_lo_controller = None

        if args.capture_l7:
            try:
                from capture import L7CaptureController, configure_client_proxy
                l7_controller = L7CaptureController(
                    capture_dir=args.capture_l7_dir,
                    proxy_port=args.capture_l7_proxy_port,
                    web_port=args.capture_l7_web_port
                )
                filter_hosts = None
                if args.capture_l7_hosts:
                    filter_hosts = [
                        h.strip() for h in args.capture_l7_hosts.split(",") if h.strip()
                    ]
                if l7_controller.start(filter_hosts=filter_hosts) is None:
                    logger.error(
                        "L7 capture failed to start; aborting run so the "
                        "client doesn't try to use a non-listening proxy."
                    )
                    l7_controller = None
                    raise SystemExit(2)
                configure_client_proxy(f"http://localhost:{args.capture_l7_proxy_port}")
            except SystemExit:
                raise
            except Exception as e:
                logger.warning(f"Failed to start L7 capture: {e}")
                l7_controller = None

        if args.scenario == "all":
            # Run full test matrix
            results = orchestrator.run_test_matrix(
                runs_per_experiment=args.runs,
                run_timeout=args.run_timeout,
                stop_on_error=args.stop_on_error,
                resume=args.resume)
            orchestrator.generate_report(results["metrics"], args.report)
        elif args.scenario:
            # Run single scenario
            results = orchestrator.run_experiment(
                scenario_name=args.scenario,
                profile_name=args.profile,
                runs=args.runs,
                ingress_profile=args.ingress_profile,
                run_timeout=args.run_timeout,
                stop_on_error=args.stop_on_error,
                resume=args.resume)

            # Compute and display metrics
            records = [
                asdict(record)
                for result in results
                for record in result.log_records
            ]
            if not records:
                records = [
                    {
                        "latency_sec": r.total_latency_sec,
                        "request_bytes": r.total_request_bytes,
                        "response_bytes": r.total_response_bytes,
                        "tokens_in": r.total_tokens_in,
                        "tokens_out": r.total_tokens_out,
                        "success": r.success,
                        "tool_calls_count": r.tool_calls_count,
                        "tool_latency_sec": (
                            r.tool_total_latency_sec / r.tool_calls_count
                            if r.tool_calls_count else 0.0
                        ),
                        "session_id": r.session_id,
                    }
                    for r in results
                ]
            metrics_defaults = orchestrator.scenarios_config.get("defaults", {})
            metrics = MetricsCalculator.calculate(
                records,
                args.scenario,
                args.profile,
                stall_gap_sec=metrics_defaults.get("stall_gap_sec"),
                burst_gap_sec=metrics_defaults.get("burst_gap_sec"))

            print("\n" + "="*60)
            print(f"Results for {args.scenario} with {args.profile}")
            print("="*60)
            print(f"  Samples: {metrics.sample_count}")
            print(f"  Latency (mean): {metrics.latency_mean:.3f}s")
            print(f"  Latency (p95):  {metrics.latency_p95:.3f}s")
            print(f"  Success rate:   {metrics.success_rate:.1f}%")
            print(f"  Request bytes:  {metrics.request_bytes_mean:.0f}")
            print(f"  Response bytes: {metrics.response_bytes_mean:.0f}")
            print(f"  UL/DL ratio:    {metrics.ul_dl_ratio_mean:.3f}")
            if metrics.ttft_p95 is not None:
                print(f"  TTFT (p95):     {metrics.ttft_p95:.3f}s")
            if metrics.ttlt_p95 is not None:
                print(f"  TTLT (p95):     {metrics.ttlt_p95:.3f}s")
            if metrics.stall_rate is not None:
                print(f"  Stall rate:     {metrics.stall_rate:.3f}")
            if metrics.burst_peak_to_mean is not None:
                print(f"  Burst P/M:      {metrics.burst_peak_to_mean:.2f}")
            if metrics.burst_cv is not None:
                print(f"  Burst CV:       {metrics.burst_cv:.2f}")
            if metrics.tool_calls_mean > 0:
                print(f"  Tool calls:     {metrics.tool_calls_mean:.1f}")
                print(f"  Loop factor:    {metrics.loop_factor:.1f}")
            if metrics.error_breakdown and any(metrics.error_breakdown.values()):
                print(f"  Errors:         {metrics.error_breakdown}")
        else:
            parser.print_help()
    finally:
        if l7_controller:
            try:
                from capture import clear_client_proxy
                l7_controller.stop()
                clear_client_proxy()
            except Exception as e:
                logger.warning(f"Failed to stop L7 capture cleanly: {e}")

        if pcap_controller:
            try:
                pcap_stop_time = time.time()
                pcap_path = pcap_controller.stop()
                if pcap_path:
                    capture_stats = pcap_controller.get_capture_stats(pcap_path)
                    capture_duration = None
                    if pcap_start_time is not None:
                        capture_duration = pcap_stop_time - pcap_start_time

                    scenario_id = args.scenario if args.scenario else "unknown"
                    if args.scenario == "all":
                        scenario_id = "test_matrix"

                    network_profile = args.profile if args.scenario and args.scenario != "all" else "multiple"

                    pcap_size = 0
                    try:
                        pcap_size = pcap_path.stat().st_size
                    except Exception:
                        pass

                    capture_metadata = {
                        "type": "pcap_capture",
                        "pcap_file": str(pcap_path),
                        "capture_dir": args.capture_dir,
                        "capture_filter": args.capture_filter,
                        "capture_duration_sec": capture_duration,
                        "capture_stats": capture_stats,
                        "interface": orchestrator.network_interface,
                        "run_id": run_id,
                    }

                    capture_record = LogRecord(
                        timestamp=time.time(),
                        scenario_id=scenario_id,
                        session_id=f"pcap_{int(pcap_start_time or time.time())}",
                        turn_index=-2,
                        run_index=-1,
                        provider="tcpdump",
                        model="",
                        request_bytes=0,
                        response_bytes=pcap_size,
                        request_bytes_source="pcap_wire",
                        response_bytes_source="pcap_wire",
                        t_request_start=pcap_start_time or 0.0,
                        latency_sec=capture_duration or 0.0,
                        network_profile=network_profile,
                        http_status=200,
                        success=True,
                        metadata=json.dumps(capture_metadata))
                    orchestrator.logger.log(capture_record)
            except Exception as e:
                logger.warning(f"Failed to stop tcpdump capture cleanly: {e}")

        if pcap_lo_controller:
            try:
                pcap_lo_stop_time = time.time()
                pcap_lo_path = pcap_lo_controller.stop()
                if pcap_lo_path:
                    lo_stats = pcap_lo_controller.get_capture_stats(pcap_lo_path)
                    lo_duration = None
                    if pcap_start_time is not None:
                        lo_duration = pcap_lo_stop_time - pcap_start_time

                    scenario_id_lo = args.scenario if args.scenario else "unknown"
                    if args.scenario == "all":
                        scenario_id_lo = "test_matrix"

                    network_profile_lo = (
                        args.profile if args.scenario and args.scenario != "all" else "multiple"
                    )

                    pcap_lo_size = 0
                    try:
                        pcap_lo_size = pcap_lo_path.stat().st_size
                    except Exception:
                        pass

                    lo_metadata = {
                        "type": "pcap_capture",
                        "pcap_file": str(pcap_lo_path),
                        "capture_dir": args.capture_dir,
                        "capture_filter": args.capture_loopback_filter or "not port 22 and not port 53",
                        "capture_duration_sec": lo_duration,
                        "capture_stats": lo_stats,
                        "interface": "lo",
                        "run_id": run_id,
                    }

                    lo_record = LogRecord(
                        timestamp=time.time(),
                        scenario_id=scenario_id_lo,
                        session_id=f"pcap_lo_{int(pcap_start_time or time.time())}",
                        turn_index=-2,
                        run_index=-1,
                        provider="tcpdump",
                        model="",
                        request_bytes=0,
                        response_bytes=pcap_lo_size,
                        request_bytes_source="pcap_wire",
                        response_bytes_source="pcap_wire",
                        t_request_start=pcap_start_time or 0.0,
                        latency_sec=lo_duration or 0.0,
                        network_profile=network_profile_lo,
                        http_status=200,
                        success=True,
                        metadata=json.dumps(lo_metadata))
                    orchestrator.logger.log(lo_record)
            except Exception as e:
                logger.warning(f"Failed to stop loopback tcpdump capture cleanly: {e}")


if __name__ == "__main__":
    main()
