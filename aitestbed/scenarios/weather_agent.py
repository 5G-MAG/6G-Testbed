"""
Weather / Environment Agent Scenario for the 6G AI Traffic Testbed.

Implements environmental awareness via a custom Open-Meteo MCP server.
Maps to TR 22.870 use cases: 6.21, 6.47, 6.51.
"""

from .base import ScenarioResult
from .agent import BaseAgentScenario


class WeatherAgentScenario(BaseAgentScenario):
    """
    Weather/environment agent scenario using the Open-Meteo MCP server.

    Uses the weather MCP server to:
    - Get current weather conditions for coordinates
    - Retrieve multi-day forecasts
    - Geocode location names to coordinates
    - Query air quality data
    """

    def __init__(self, client, logger, config):
        config.setdefault("server_group", "weather")
        super().__init__(client, logger, config)

    @property
    def scenario_type(self) -> str:
        return "weather_agent"

    async def run_async(
        self,
        network_profile: str,
        run_index: int = 0,
    ) -> ScenarioResult:
        session_id = self._create_session_id()
        model = self.config.get("model", "gpt-5-mini")
        prompts = self.config.get("prompts", [
            "Get the current weather and 48-hour forecast for Paris, France. "
            "Assess outdoor safety risks including temperature, precipitation, "
            "wind, and UV index."
        ])

        result = ScenarioResult(
            scenario_id=self.scenario_id,
            session_id=session_id,
            network_profile=network_profile,
            run_index=run_index,
        )

        system_prompt = self.config.get("system_prompt", DEFAULT_WEATHER_SYSTEM_PROMPT)

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


class NavigationWeatherAgentScenario(BaseAgentScenario):
    """
    Combined navigation + weather agent for route weather assessment.

    Uses both Google Maps and weather tools to evaluate weather
    conditions along a travel route (TR 22.870 UC 6.51).
    """

    def __init__(self, client, logger, config):
        config.setdefault("server_group", "navigation")
        super().__init__(client, logger, config)

    @property
    def scenario_type(self) -> str:
        return "navigation_weather_agent"

    async def run_async(
        self,
        network_profile: str,
        run_index: int = 0,
    ) -> ScenarioResult:
        session_id = self._create_session_id()
        model = self.config.get("model", "gpt-5-mini")
        prompts = self.config.get("prompts", [
            "Plan a driving route from Munich to Vienna. Check weather at the "
            "origin, a midpoint waypoint, and the destination. Flag any "
            "sections with adverse conditions."
        ])

        result = ScenarioResult(
            scenario_id=self.scenario_id,
            session_id=session_id,
            network_profile=network_profile,
            run_index=run_index,
        )

        system_prompt = self.config.get(
            "system_prompt", DEFAULT_NAVIGATION_WEATHER_SYSTEM_PROMPT
        )

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


DEFAULT_WEATHER_SYSTEM_PROMPT = """\
You are an environmental awareness assistant with access to weather tools.
Use them to provide weather analysis, safety assessments, and forecasts.

Available tools:
- get_current_weather: Get current conditions (temp, wind, humidity, etc.)
- get_forecast: Get hourly or daily forecast for a location
- geocode_location: Convert a place name to latitude/longitude coordinates
- get_air_quality: Get current air quality index and pollutant levels

Assessment workflow:
1. Geocode any named locations to get coordinates
2. Get current weather conditions
3. Get the forecast for the requested time period
4. Assess safety risks (extreme temps, storms, high winds, poor air quality)
5. Provide clear recommendations with specific numbers

Always include temperature, precipitation probability, wind speed, and \
any relevant warnings in your response."""

DEFAULT_NAVIGATION_WEATHER_SYSTEM_PROMPT = """\
You are a navigation and weather assistant with access to both Google Maps \
and weather tools. Use them together to provide weather-aware route planning.

Maps tools:
- maps_geocode, maps_directions, maps_search_places, maps_distance_matrix

Weather tools:
- get_current_weather, get_forecast, geocode_location, get_air_quality

Route weather workflow:
1. Use maps_directions to plan the route and identify waypoints
2. Get weather at origin, key waypoints, and destination
3. Flag any route segments with adverse conditions (rain, snow, fog, high winds)
4. Suggest timing adjustments or alternative routes if needed
5. Provide a clear route summary with weather overlay

Always include driving time, distance, and weather conditions at each waypoint."""
