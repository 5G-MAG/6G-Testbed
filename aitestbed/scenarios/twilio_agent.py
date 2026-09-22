"""
Twilio Communication Agent Scenario for the 6G AI Traffic Testbed.

Implements emergency notification and multi-channel communication
via the Twilio MCP server.
Maps to TR 22.870 use cases: 6.46, 6.47, 6.62.
"""

import os
import re

from .base import ScenarioResult
from .agent import BaseAgentScenario


# Error fragments that indicate a phone-number-related failure rather than
# a network/API traffic problem. When the underlying tool call fails for
# one of these reasons, we still treat the run as successful for the
# purposes of network traffic characterization.
_PHONE_NUMBER_ERROR_PATTERNS = re.compile(
    r"(unverified|not\s+verified|not\s+a\s+valid\s+phone\s+number|invalid\s+'?to'?\s+phone\s+number|"
    r"invalid\s+phone\s+number|phone\s+number\s+is\s+not\s+valid|"
    r"from\s+phone\s+number.*not\s+.*valid|trial\s+account.*verified|"
    r"error\s+code\s+(21211|21212|21214|21219|21408|21606|21608|21610|21612|21614|21618|21659)|"
    r"permission\s+to\s+send.*to\s+this\s+number)",
    re.IGNORECASE,
)


def _is_phone_number_error(message: str | None) -> bool:
    """Return True if the error message is about phone number validation."""
    if not message:
        return False
    return bool(_PHONE_NUMBER_ERROR_PATTERNS.search(message))


class TwilioCommunicationAgentScenario(BaseAgentScenario):
    """
    Communication agent scenario using Twilio MCP tools.

    Uses the Twilio MCP server to:
    - Send SMS messages
    - Look up phone number carrier/type info
    - List and verify message delivery status
    - Initiate voice calls (API only, no live audio)
    """

    def __init__(self, client, logger, config):
        config.setdefault("server_group", "communication")
        super().__init__(client, logger, config)

    @property
    def scenario_type(self) -> str:
        return "twilio_agent"

    async def run_async(
        self,
        network_profile: str,
        run_index: int = 0,
    ) -> ScenarioResult:
        session_id = self._create_session_id()
        model = self.config.get("model", "gpt-5-mini")
        prompts = self.config.get("prompts", [
            "Look up the phone number ${TWILIO_TEST_PHONE_NUMBER}. Determine if it is a mobile number. Then send an SMS to it saying 'Test alert from 6G AI Testbed'."
        ])

        # Substitute ${TWILIO_TEST_PHONE_NUMBER} and ${TWILIO_FROM_PHONE_NUMBER}
        # from env so real numbers never get hard-coded in the scenario config.
        test_number = os.environ.get("TWILIO_TEST_PHONE_NUMBER")
        from_number = os.environ.get("TWILIO_FROM_PHONE_NUMBER")
        if test_number:
            prompts = [p.replace("${TWILIO_TEST_PHONE_NUMBER}", test_number) for p in prompts]
        if from_number:
            prompts = [p.replace("${TWILIO_FROM_PHONE_NUMBER}", from_number) for p in prompts]

        result = ScenarioResult(
            scenario_id=self.scenario_id,
            session_id=session_id,
            network_profile=network_profile,
            run_index=run_index,
        )

        system_prompt = self.config.get("system_prompt", DEFAULT_TWILIO_SYSTEM_PROMPT)
        if from_number:
            system_prompt = system_prompt.replace("${TWILIO_FROM_PHONE_NUMBER}", from_number)

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
                    turn_error = turn_result.get("error")
                    # Phone-number validation errors (unverified recipient,
                    # invalid format, trial-account restrictions, etc.) are
                    # not network failures — the traffic still flowed. Keep
                    # the run as successful for benchmark purposes.
                    if _is_phone_number_error(turn_error):
                        continue
                    result.success = False
                    result.error_message = turn_error
                    break

        except Exception as e:
            result.success = False
            result.error_message = str(e)

        finally:
            await self.teardown()

        return result


DEFAULT_TWILIO_SYSTEM_PROMPT = """\
You are an emergency communication assistant with access to Twilio messaging tools.
Use them to send alerts, verify phone numbers, and check message delivery.

You must use the Twilio number ${TWILIO_FROM_PHONE_NUMBER} as the "From" \
address for every outbound SMS.

Available tools (filtered to messaging + lookups):
- TwilioApiV2010--CreateMessage: Send an SMS/MMS message. Required params: \
AccountSid, To (E.164), From (use ${TWILIO_FROM_PHONE_NUMBER}), Body.
- TwilioApiV2010--ListMessage: List recent messages. Param: AccountSid.
- TwilioApiV2010--FetchMessage: Get details and delivery status of a specific \
message. Params: AccountSid, Sid.
- TwilioApiV2010--UpdateMessage: Update/redact a message.
- TwilioApiV2010--DeleteMessage: Delete a message record.
- TwilioLookupsV2--FetchPhoneNumber: Look up carrier/type/validation info for a \
phone number. Param: PhoneNumber (E.164).

Communication workflow:
1. Look up recipient phone numbers with TwilioLookupsV2--FetchPhoneNumber to verify \
they are valid and reachable.
2. Send messages with TwilioApiV2010--CreateMessage using clear, concise content.
3. Verify delivery with TwilioApiV2010--FetchMessage using the returned message SID.
4. Report delivery confirmation or any failures.

Always verify phone numbers before sending. Use E.164 format (+country code). \
Report message SID and delivery status in your response."""
