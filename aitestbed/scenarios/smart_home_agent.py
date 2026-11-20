"""
Smart Home / IoT Agent Scenario for the 6G AI Traffic Testbed.

Implements smart home device coordination via Home Assistant MCP server.
Maps to TR 22.870 use cases: 6.8, 6.45, 6.46.
"""

from .base import ScenarioResult
from .agent import BaseAgentScenario


class SmartHomeAgentScenario(BaseAgentScenario):
    """
    Smart home agent scenario using Home Assistant MCP tools.

    Uses the Home Assistant MCP server to:
    - List and query device/entity states
    - Control devices (lights, locks, thermostats, etc.)
    - Execute automation scenes
    - Query sensor history
    """

    def __init__(self, client, logger, config):
        config.setdefault("server_group", "smart_home")
        super().__init__(client, logger, config)

    @property
    def scenario_type(self) -> str:
        return "smart_home_agent"

    async def run_async(
        self,
        network_profile: str,
        run_index: int = 0,
    ) -> ScenarioResult:
        session_id = self._create_session_id()
        model = self.config.get("model", "gpt-5-mini")
        prompts = self.config.get("prompts", [
            "List all available devices. Check the status of any motion sensors and temperature sensors, then summarize the current home state."
        ])

        result = ScenarioResult(
            scenario_id=self.scenario_id,
            session_id=session_id,
            network_profile=network_profile,
            run_index=run_index,
        )

        system_prompt = self.config.get("system_prompt", DEFAULT_SMART_HOME_SYSTEM_PROMPT)

        try:
            await self.setup()
            self._emit_discovery_records(result, session_id, run_index, network_profile)

            for prompt_index, user_prompt in enumerate(prompts):
                turn_result = await self._run_agent_turn(
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                    model=model,
                    session_id=session_id,
                    turn_index=prompt_index,
                    run_index=run_index,
                    network_profile=network_profile,
                )

                result.turn_count += 1
                result.api_call_count += turn_result["api_calls"]
                result.tool_calls_count += turn_result["tool_calls"]
                result.tool_total_latency_sec += turn_result["tool_latency"]
                result.total_latency_sec += turn_result["total_latency"]
                result.total_request_bytes += turn_result["request_bytes"]
                result.total_response_bytes += turn_result["response_bytes"]
                result.total_tokens_in += turn_result.get("tokens_in", 0)
                result.total_tokens_out += turn_result.get("tokens_out", 0)
                result.log_records.extend(turn_result["log_records"])

                if not turn_result["success"]:
                    result.success = False
                    result.error_message = turn_result.get("error")
                    break

        except Exception as e:
            result.success = False
            result.error_message = str(e)

        finally:
            await self.teardown()

        return result


DEFAULT_SMART_HOME_SYSTEM_PROMPT = """\
You are a smart home assistant with access to Home Assistant tools.
Use them to monitor, control, and coordinate smart home devices.

Available tools:
- list_entities: List all available devices and entities with their domains
- get_entity_state: Get the current state and attributes of a specific entity
- call_service: Call a Home Assistant service (e.g. turn on light, lock door)
- get_history: Get recent state history for an entity
- fire_event: Fire a custom event in Home Assistant

Device coordination workflow:
1. List entities to discover available devices
2. Query states of relevant sensors and devices
3. Take actions based on the current state (e.g. turn on lights if dark)
4. Verify actions by re-checking entity states
5. Report a clear summary of the home state and any actions taken

Always check device state before and after actions. Report specific \
entity IDs, states, and any changes made."""
