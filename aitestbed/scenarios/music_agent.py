"""
Music Agent Scenario for the 6G AI Traffic Testbed.

Implements music search and playlist composition via Spotify MCP tools.
"""

import json

from .base import BaseScenario, ScenarioResult
from .agent import BaseAgentScenario


class MusicAgentScenario(BaseAgentScenario):
    """
    Music agent scenario using Spotify MCP tools.

    Uses the Spotify MCP server to:
    - Search for tracks, artists, and albums
    - Get personalized recommendations
    - Compose playlists based on user criteria
    """

    def __init__(self, client, logger, config):
        config.setdefault("server_group", "music")
        super().__init__(client, logger, config)

    @property
    def scenario_type(self) -> str:
        return "music_agent"

    async def run_async(
        self,
        network_profile: str,
        run_index: int = 0,
    ) -> ScenarioResult:
        session_id = self._create_session_id()
        model = self.config.get("model", "gpt-5-mini")
        prompts = self.config.get("prompts", [
            "Find me upbeat pop songs for a workout playlist with at least 10 tracks."
        ])

        result = ScenarioResult(
            scenario_id=self.scenario_id,
            session_id=session_id,
            network_profile=network_profile,
            run_index=run_index,
        )

        system_prompt = self.config.get("system_prompt", DEFAULT_MUSIC_SYSTEM_PROMPT)

        try:
            await self.setup()

            for prompt_index, user_prompt in enumerate(prompts):
                await self._wait_between_prompts_async(prompt_index)
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


class MusicResearchAgentScenario(BaseAgentScenario):
    """
    Music research agent combining Spotify tools with web search.

    Multi-hop scenario: searches Spotify for music, then uses web search
    to find reviews, concert info, or background on artists.
    """

    def __init__(self, client, logger, config):
        config.setdefault("server_group", "music_research")
        super().__init__(client, logger, config)

    @property
    def scenario_type(self) -> str:
        return "music_research_agent"

    async def run_async(
        self,
        network_profile: str,
        run_index: int = 0,
    ) -> ScenarioResult:
        session_id = self._create_session_id()
        model = self.config.get("model", "gpt-5-mini")
        prompts = self.config.get("prompts", [
            "Research the top jazz albums of 2025 and create a playlist of standout tracks with background on each artist."
        ])

        result = ScenarioResult(
            scenario_id=self.scenario_id,
            session_id=session_id,
            network_profile=network_profile,
            run_index=run_index,
        )

        system_prompt = self.config.get(
            "system_prompt", DEFAULT_MUSIC_RESEARCH_SYSTEM_PROMPT
        )

        try:
            await self.setup()

            for prompt_index, user_prompt in enumerate(prompts):
                await self._wait_between_prompts_async(prompt_index)
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


DEFAULT_MUSIC_SYSTEM_PROMPT = """\
You are a music assistant with access to Spotify tools. Use them to help the user.

Available tools:
- spotify_search_tracks: Search for songs by name, artist, or genre
- spotify_search_artists: Find artists
- spotify_search_albums: Find albums
- spotify_search_playlists: Find curated playlists by mood, genre, or activity
- spotify_artist_albums: Get an artist's albums and singles (by artist ID)
- spotify_album_tracks: Get the track listing for an album (by album ID)

When building playlists:
1. Search for relevant tracks, artists, or playlists
2. Use artist IDs to browse their albums, then get album tracks
3. Search playlists for mood/genre-based discovery
4. Compile a cohesive playlist with song name, artist, and Spotify link

Be creative and thorough. Explain your choices."""

DEFAULT_MUSIC_RESEARCH_SYSTEM_PROMPT = """\
You are a music research assistant with access to Spotify and web search tools.

Available tools:
- spotify_search_tracks, spotify_search_artists, spotify_search_albums
- spotify_search_playlists, spotify_artist_albums, spotify_album_tracks
- brave_web_search: Search the web for reviews, articles, concert info
- fetch: Retrieve content from web pages

Research workflow:
1. Use Spotify tools to find music (tracks, artists, albums, playlists)
2. Use web search to find reviews, background info, or concert schedules
3. Fetch relevant articles for deeper analysis
4. Synthesize findings into a well-organized response with Spotify links and sources

Be thorough and cite your sources."""
