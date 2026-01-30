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
from pathlib import Path
from typing import Optional
from datetime import datetime
from dataclasses import asdict

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()  # Loads from .env in current directory
load_dotenv(Path(__file__).parent / ".env")  # Also try testbed/.env

from clients import OpenAIClient, GeminiClient, DeepSeekClient, VLLMClient
from analysis import TrafficLogger, MetricsCalculator, LogRecord
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
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("orchestrator")


class TestbedOrchestrator:
    """
    Main orchestrator for running traffic characterization experiments.
    """

    def __init__(
        self,
        config_path: str = "configs/scenarios.yaml",
        profiles_path: str = "configs/profiles.yaml",
        db_path: str = "logs/traffic_logs.db",
        network_interface: str = "eth0",
        egress_only: bool = False
    ):
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

        # Initialize components
        self.logger = TrafficLogger(db_path)
        self.emulator = NetworkEmulator(
            interface=network_interface,
            profiles_path=str(self.profiles_path),
            bidirectional=not egress_only
        )

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
            else:
                raise ValueError(f"Unknown provider: {provider}")
        return self._clients[provider]

    def run_experiment(
        self,
        scenario_name: str,
        profile_name: str,
        runs: int = 10,
        inter_run_delay: float = 1.0,
        ingress_profile: Optional[str] = None
    ) -> list[ScenarioResult]:
        """
        Run a single experiment (scenario + profile combination).

        Args:
            scenario_name: Name of the scenario to run
            profile_name: Network profile to apply (egress, and ingress if not specified separately)
            runs: Number of runs
            inter_run_delay: Delay between runs in seconds
            ingress_profile: Optional separate profile for ingress traffic.
                            None = use same as egress, "none" = no ingress shaping

        Returns:
            List of ScenarioResult objects
        """
        # Get scenario configuration
        scenario_config = self.scenarios_config["scenarios"].get(scenario_name)
        if not scenario_config:
            raise ValueError(f"Unknown scenario: {scenario_name}")

        # Determine scenario type and class
        scenario_type = scenario_config.get("type", "chat")
        scenario_class = self.scenario_classes.get(scenario_type)
        if not scenario_class:
            raise ValueError(f"Unknown scenario type: {scenario_type}")

        # Get provider and client
        provider = scenario_config.get("provider", "openai")
        client = self.get_client(provider)

        # Create scenario instance
        scenario_config["scenario_id"] = scenario_name
        scenario = scenario_class(client, self.logger, scenario_config)

        # Apply network profile
        if ingress_profile:
            logger.info(f"Applying network profile: egress={profile_name}, ingress={ingress_profile}")
        else:
            logger.info(f"Applying network profile: {profile_name} (bidirectional)")
        if not self.emulator.apply_profile(profile_name, ingress_profile=ingress_profile):
            logger.warning(f"Failed to apply network profile (may require sudo)")

        results = []
        defaults = self.scenarios_config.get("defaults", {})
        retry_count = int(defaults.get("retry_count", 0))
        retry_backoff_sec = float(defaults.get("retry_backoff_sec", 1.0))
        retry_backoff_multiplier = float(defaults.get("retry_backoff_multiplier", 2.0))

        try:
            for run_index in range(runs):
                logger.info(f"Running {scenario_name} [{run_index + 1}/{runs}] with profile {profile_name}")

                attempt = 0
                retry_reason = None
                while True:
                    t_start = time.time()
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

        return results

    def _get_retry_reason(self, result: ScenarioResult) -> Optional[str]:
        """Return a retry reason for transient failures."""
        if not result.log_records:
            error_text = (result.error_message or "").lower()
            if "timeout" in error_text or "timed out" in error_text:
                return "timeout"
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

            error_type = (record.error_type or "").lower()
            if "timeout" in error_type or "timed out" in error_type:
                return "timeout"

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
        runs_per_experiment: int = 10
    ) -> dict:
        """
        Run a full test matrix.

        Args:
            matrix: Test matrix definition (uses config if not provided)
            runs_per_experiment: Default runs per experiment

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
                        runs=runs
                    )

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
                        burst_gap_sec=metrics_defaults.get("burst_gap_sec"),
                    )
                    all_metrics.append(metrics)

                    logger.info(f"Metrics: latency_mean={metrics.latency_mean:.3f}s, "
                               f"success_rate={metrics.success_rate:.1f}%")

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
        output_path: str = "reports/experiment_report.json"
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
        default="ideal_6g",
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
        default="configs/scenarios.yaml",
        help="Path to scenarios config"
    )
    parser.add_argument(
        "--profiles",
        default="configs/profiles.yaml",
        help="Path to network profiles config"
    )
    parser.add_argument(
        "--db",
        default="logs/traffic_logs.db",
        help="Path to SQLite database"
    )
    parser.add_argument(
        "--interface",
        default="eth0",
        help="Network interface for emulation"
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
        default="reports/experiment_report.json",
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
        default=None,
        help="Optional tcpdump filter expression (e.g., 'port 443')"
    )
    parser.add_argument(
        "--capture-dir",
        default="capture/captures",
        help="Directory for pcap captures"
    )
    parser.add_argument(
        "--capture-l7",
        action="store_true",
        help="Enable L7 capture using mitmproxy"
    )
    parser.add_argument(
        "--capture-l7-dir",
        default="capture/l7_captures",
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

    args = parser.parse_args()

    orchestrator = TestbedOrchestrator(
        config_path=args.config,
        profiles_path=args.profiles,
        db_path=args.db,
        network_interface=args.interface,
        egress_only=args.egress_only
    )

    if args.list_scenarios:
        print("\nAvailable scenarios:")
        for name, config in orchestrator.scenarios_config["scenarios"].items():
            print(f"  - {name}: {config.get('description', 'No description')}")
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
    try:
        if args.capture_pcap:
            try:
                from capture import CaptureController
                pcap_controller = CaptureController(
                    interface=args.interface,
                    capture_dir=args.capture_dir
                )
                pcap_start_time = time.time()
                pcap_file = pcap_controller.start(filter_expr=args.capture_filter)
                if not pcap_file:
                    logger.warning("Failed to start tcpdump capture")
                    pcap_start_time = None
            except Exception as e:
                logger.warning(f"Failed to start tcpdump capture: {e}")
                pcap_controller = None
                pcap_start_time = None

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
                l7_controller.start(filter_hosts=filter_hosts)
                configure_client_proxy(f"http://localhost:{args.capture_l7_proxy_port}")
            except Exception as e:
                logger.warning(f"Failed to start L7 capture: {e}")
                l7_controller = None

        if args.scenario == "all":
            # Run full test matrix
            results = orchestrator.run_test_matrix(runs_per_experiment=args.runs)
            orchestrator.generate_report(results["metrics"], args.report)
        elif args.scenario:
            # Run single scenario
            results = orchestrator.run_experiment(
                scenario_name=args.scenario,
                profile_name=args.profile,
                runs=args.runs,
                ingress_profile=args.ingress_profile
            )

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
                burst_gap_sec=metrics_defaults.get("burst_gap_sec"),
            )

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
                        t_request_start=pcap_start_time or 0.0,
                        latency_sec=capture_duration or 0.0,
                        network_profile=network_profile,
                        http_status=200,
                        success=True,
                        metadata=json.dumps(capture_metadata),
                    )
                    orchestrator.logger.log(capture_record)
            except Exception as e:
                logger.warning(f"Failed to stop tcpdump capture cleanly: {e}")


if __name__ == "__main__":
    main()
