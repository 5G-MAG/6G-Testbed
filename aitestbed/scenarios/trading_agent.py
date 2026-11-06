"""
Trading / Market Data Agent Scenario for the 6G AI Traffic Testbed.

Implements financial market data analysis via the Alpaca MCP server.
Uses market data API only (no trading/account operations).
"""

from .base import ScenarioResult
from .agent import BaseAgentScenario


class TradingAgentScenario(BaseAgentScenario):
    """
    Market data agent scenario using Alpaca MCP tools.

    Uses the Alpaca MCP server (market data API) to:
    - Retrieve stock/crypto quotes, bars, and snapshots
    - Look up option contracts and quotes
    - Check market calendar and clock
    - Look up asset information

    NOTE: This scenario uses a market-data-only API key.
    No account, portfolio, order, or position tools are available.
    """

    def __init__(self, client, logger, config):
        config.setdefault("server_group", "trading")
        super().__init__(client, logger, config)

    @property
    def scenario_type(self) -> str:
        return "trading_agent"

    async def run_async(
        self,
        network_profile: str,
        run_index: int = 0,
    ) -> ScenarioResult:
        session_id = self._create_session_id()
        model = self.config.get("model", "gpt-5-mini")
        prompts = self.config.get("prompts", [
            "Get the latest quote and a 5-day bar chart for AAPL and MSFT, then compare their recent performance."
        ])

        result = ScenarioResult(
            scenario_id=self.scenario_id,
            session_id=session_id,
            network_profile=network_profile,
            run_index=run_index,
        )

        system_prompt = self.config.get("system_prompt", DEFAULT_TRADING_SYSTEM_PROMPT)

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


DEFAULT_TRADING_SYSTEM_PROMPT = """\
You are a financial market data analyst with access to Alpaca market data tools.
Use them to retrieve and analyze stock, crypto, and options data.

Available tools:
- get_stock_latest_quote: Get the latest bid/ask quote for a stock symbol
- get_stock_latest_trade: Get the most recent trade for a stock
- get_stock_bars: Get OHLCV bars (historical price data) for a stock
- get_stock_snapshot: Get a full snapshot (quote, trade, bar) for a stock
- get_crypto_latest_quote: Get the latest quote for a crypto pair
- get_crypto_bars: Get historical OHLCV bars for a crypto pair
- get_crypto_snapshot: Get a full snapshot for a crypto pair
- get_option_contracts: Search for available option contracts
- get_option_latest_quote: Get the latest quote for an option contract
- get_asset: Look up details about a specific asset by symbol
- get_all_assets: List all available assets
- get_calendar: Get the market calendar (trading days, open/close times)
- get_clock: Check if the market is currently open

Analysis workflow:
1. Use get_clock / get_calendar to check market status
2. Retrieve quotes and bars for the requested symbols
3. Compare metrics across assets (price, volume, spread)
4. Provide clear analysis with specific numbers

Be precise with numbers. Always state the data timestamp so the user knows \
how fresh the data is."""
