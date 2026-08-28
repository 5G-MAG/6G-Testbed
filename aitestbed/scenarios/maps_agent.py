"""
Maps / Navigation Agent Scenario for the 6G AI Traffic Testbed.

Implements location-aware navigation and routing via Google Maps MCP server.
Maps to TR 22.870 use cases: 6.6, 6.9, 6.21, 6.44, 6.47, 6.51.
"""

from .base import ScenarioResult
from .agent import BaseAgentScenario


class MapsAgentScenario(BaseAgentScenario):
    """
    Navigation agent scenario using Google Maps MCP tools.

    Uses the Google Maps MCP server to:
    - Geocode addresses and coordinates
    - Search for nearby places (POIs)
    - Calculate routes and directions
    - Build distance matrices between locations
    - Query elevation data
    """

    def __init__(self, client, logger, config):
        config.setdefault("server_group", "maps")
        super().__init__(client, logger, config)

    @property
    def scenario_type(self) -> str:
        return "maps_agent"

    async def run_async(
        self,
        network_profile: str,
        run_index: int = 0,
    ) -> ScenarioResult:
        session_id = self._create_session_id()
        model = self.config.get("model", "gpt-5-mini")
        prompts = self.config.get("prompts", [
            "Plan a route from Berlin to Munich via Nuremberg. Compare driving time and distance, and suggest a rest stop near the midpoint."
        ])

        result = ScenarioResult(
            scenario_id=self.scenario_id,
            session_id=session_id,
            network_profile=network_profile,
            run_index=run_index,
        )

        system_prompt = self.config.get("system_prompt", DEFAULT_MAPS_SYSTEM_PROMPT)

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


DEFAULT_MAPS_SYSTEM_PROMPT = """\
You are a navigation and location intelligence assistant with access to Google Maps tools.
Use them to help users with routing, place discovery, and geospatial analysis.

Available tools:
- maps_geocode: Convert an address or place name to coordinates
- maps_reverse_geocode: Convert coordinates to a human-readable address
- maps_search_places: Search for nearby places (restaurants, EV chargers, etc.)
- maps_place_details: Get detailed info about a specific place (hours, rating, etc.)
- maps_distance_matrix: Calculate travel time/distance between multiple origins and destinations
- maps_directions: Get turn-by-turn route between two points
- maps_elevation: Get elevation data for coordinates

Analysis workflow:
1. Geocode any named locations to get coordinates
2. Use directions or distance_matrix for routing analysis
3. Search for relevant places along routes or near points of interest
4. Get place details for specific recommendations
5. Provide clear analysis with distances, durations, and specific place recommendations

Always include travel times and distances in your responses."""
