"""
Playwright Agent Scenario for the 6G AI Traffic Testbed.

Implements browser automation testing via Playwright MCP tools,
capturing the traffic pattern of agentic web UI testing.
"""

from .base import ScenarioResult
from .agent import BaseAgentScenario


class PlaywrightAgentScenario(BaseAgentScenario):
    """
    Browser automation agent scenario using Playwright MCP tools.

    Uses the Playwright MCP server to:
    - Navigate to web pages
    - Take screenshots for visual verification
    - Click elements and fill forms
    - Extract text content from pages
    - Run JavaScript in page context
    """

    def __init__(self, client, logger, config):
        config.setdefault("server_group", "playwright")
        super().__init__(client, logger, config)
        self.use_screenshots = config.get("use_screenshots", True)

    @property
    def scenario_type(self) -> str:
        return "playwright_agent"

    async def run_async(
        self,
        network_profile: str,
        run_index: int = 0,
    ) -> ScenarioResult:
        session_id = self._create_session_id()
        model = self.config.get("model", "gpt-5-mini")
        prompts = self.config.get("prompts", [
            "Navigate to https://example.com, extract the page title and all visible text, and summarize what you see."
        ])

        result = ScenarioResult(
            scenario_id=self.scenario_id,
            session_id=session_id,
            network_profile=network_profile,
            run_index=run_index,
        )

        system_prompt = self.config.get("system_prompt", self._default_system_prompt())

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

    def _default_system_prompt(self) -> str:
        if self.use_screenshots:
            return DEFAULT_PLAYWRIGHT_SYSTEM_PROMPT
        return DEFAULT_PLAYWRIGHT_TEXT_SYSTEM_PROMPT


DEFAULT_PLAYWRIGHT_SYSTEM_PROMPT = """\
You are a browser automation assistant with access to Playwright tools. \
Use them to navigate websites, interact with page elements, and verify content.

Available tools:
- playwright_navigate: Go to a URL and get the page title, status code, and a text snippet
- playwright_screenshot: Capture a full-page or element screenshot (returns base64 PNG)
- playwright_click: Click an element by CSS selector
- playwright_fill: Fill a form field by CSS selector
- playwright_get_text: Extract visible text from the page or a specific element
- playwright_evaluate: Run JavaScript in the page context

Testing workflow:
1. Navigate to the target URL
2. Verify the page loaded correctly (check title, status code)
3. Interact with elements as needed (click, fill forms)
4. Extract text or take screenshots to verify outcomes
5. Report your findings with specific details

Be methodical and verify each step before proceeding to the next."""

DEFAULT_PLAYWRIGHT_TEXT_SYSTEM_PROMPT = """\
You are a browser automation assistant with access to Playwright tools. \
Use them to navigate websites, interact with page elements, and verify content.

Available tools:
- playwright_navigate: Go to a URL and get the page title, status code, and a text snippet
- playwright_click: Click an element by CSS selector
- playwright_fill: Fill a form field by CSS selector
- playwright_get_text: Extract visible text from the page or a specific element
- playwright_evaluate: Run JavaScript in the page context

Testing workflow:
1. Navigate to the target URL
2. Verify the page loaded correctly (check title, status code)
3. Use playwright_get_text to read page content instead of screenshots
4. Interact with elements as needed (click, fill forms)
5. Extract text to verify outcomes
6. Report your findings with specific details

Be methodical and verify each step before proceeding to the next."""
