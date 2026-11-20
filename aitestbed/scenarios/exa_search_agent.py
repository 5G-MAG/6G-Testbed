"""
Exa Neural Search Agent Scenario for the 6G AI Traffic Testbed.

Implements deep web research via Exa's neural search MCP server.
Maps to TR 22.870 use cases: 6.32 (external knowledge), 6.51 Cat.3 (knowledge DBs).
"""

from .base import ScenarioResult
from .agent import BaseAgentScenario


class ExaSearchAgentScenario(BaseAgentScenario):
    """
    Neural search agent scenario using Exa MCP tools.

    Uses the Exa MCP server to:
    - Perform semantic/neural web searches
    - Find pages similar to a given URL
    - Extract full page content for analysis
    """

    def __init__(self, client, logger, config):
        config.setdefault("server_group", "exa")
        super().__init__(client, logger, config)

    @property
    def scenario_type(self) -> str:
        return "exa_search_agent"

    async def run_async(
        self,
        network_profile: str,
        run_index: int = 0,
    ) -> ScenarioResult:
        session_id = self._create_session_id()
        model = self.config.get("model", "gpt-5-mini")
        prompts = self.config.get("prompts", [
            "Search for the latest 3GPP Release 20 AI/ML specifications published in 2026. Get the full content of the top 3 results and summarize the key requirements."
        ])

        result = ScenarioResult(
            scenario_id=self.scenario_id,
            session_id=session_id,
            network_profile=network_profile,
            run_index=run_index,
        )

        system_prompt = self.config.get("system_prompt", DEFAULT_EXA_SYSTEM_PROMPT)

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


DEFAULT_EXA_SYSTEM_PROMPT = """\
You are a deep research assistant with access to Exa neural search tools.
Use them to find high-quality, semantically relevant information.

Available tools:
- web_search_exa: Perform a neural/semantic web search (returns URLs and snippets)
- find_similar: Find web pages similar to a given URL
- get_contents: Extract the full text content of one or more URLs

Research workflow:
1. Use web_search_exa to find relevant pages for the query
2. Use get_contents to extract full text from the most promising results
3. Optionally use find_similar to discover related resources
4. Synthesize findings into a clear, well-sourced summary

Be thorough. Always cite specific URLs as sources. Prefer primary sources \
(specifications, official docs, research papers) over secondary commentary."""
