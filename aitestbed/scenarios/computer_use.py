"""
Computer Use Scenario for the 6G AI Traffic Testbed.

Implements the OpenAI Computer Use tool loop with local execution.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Optional

from openai import OpenAI

from .base import BaseScenario, ScenarioResult
from analysis.logger import LogRecord

logger = logging.getLogger(__name__)


@dataclass
class ComputerToolCall:
    """Parsed computer tool call."""
    call_id: str
    action: dict
    call_type: str


class ComputerUseExecutor:
    """
    Execute computer-use actions locally.

    Default backend uses Playwright for a headless browser.
    """

    def __init__(
        self,
        backend: str = "playwright",
        viewport: tuple[int, int] = (1280, 720),
        full_page: bool = True,
        navigation_timeout_ms: int = 30000,
        wait_after_action_ms: int = 250,
    ):
        self.backend = backend
        self.viewport = viewport
        self.full_page = full_page
        self.navigation_timeout_ms = navigation_timeout_ms
        self.wait_after_action_ms = wait_after_action_ms

        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None

    def start(self) -> None:
        """Initialize the local execution environment."""
        if self.backend != "playwright":
            raise ValueError(f"Unsupported computer backend: {self.backend}")

        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise ImportError(
                "Playwright is required for computer use. "
                "Install with: pip install playwright"
            ) from exc

        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=True)
        self._context = self._browser.new_context(
            viewport={"width": self.viewport[0], "height": self.viewport[1]}
        )
        self._page = self._context.new_page()
        self._page.goto("about:blank", timeout=self.navigation_timeout_ms)

    def close(self) -> None:
        """Shut down the local execution environment."""
        if self._page:
            try:
                self._page.close()
            except Exception:
                pass
            self._page = None
        if self._context:
            try:
                self._context.close()
            except Exception:
                pass
            self._context = None
        if self._browser:
            try:
                self._browser.close()
            except Exception:
                pass
            self._browser = None
        if self._playwright:
            try:
                self._playwright.stop()
            except Exception:
                pass
            self._playwright = None

    def execute(self, action: dict) -> dict:
        """Execute a computer action and return screenshot output."""
        if not self._page:
            raise RuntimeError("Computer executor is not started")

        action_type = (action.get("type") or action.get("action") or "").lower()
        success = True
        error = None

        try:
            if action_type in ("click", "left_click"):
                self._page.mouse.click(
                    action.get("x", 0),
                    action.get("y", 0),
                    button="left"
                )
            elif action_type == "double_click":
                self._page.mouse.click(
                    action.get("x", 0),
                    action.get("y", 0),
                    button="left",
                    click_count=2
                )
            elif action_type == "right_click":
                self._page.mouse.click(
                    action.get("x", 0),
                    action.get("y", 0),
                    button="right"
                )
            elif action_type == "move":
                self._page.mouse.move(action.get("x", 0), action.get("y", 0))
            elif action_type in ("scroll", "scroll_wheel"):
                dx = action.get("scroll_x", action.get("delta_x", 0))
                dy = action.get("scroll_y", action.get("delta_y", 0))
                self._page.mouse.wheel(dx, dy)
            elif action_type in ("type", "text"):
                text = action.get("text", "")
                self._page.keyboard.type(text)
            elif action_type in ("keypress", "key"):
                keys = action.get("keys") or action.get("key")
                if isinstance(keys, list):
                    for key in keys:
                        self._page.keyboard.press(key)
                elif keys:
                    self._page.keyboard.press(keys)
            elif action_type == "drag":
                from_x = action.get("from_x", action.get("x1", action.get("x", 0)))
                from_y = action.get("from_y", action.get("y1", action.get("y", 0)))
                to_x = action.get("to_x", action.get("x2", action.get("x", 0)))
                to_y = action.get("to_y", action.get("y2", action.get("y", 0)))
                self._page.mouse.move(from_x, from_y)
                self._page.mouse.down()
                self._page.mouse.move(to_x, to_y)
                self._page.mouse.up()
            elif action_type in ("navigate", "open_url", "goto"):
                url = action.get("url") or action.get("href") or action.get("text") or ""
                if not url:
                    raise ValueError("navigate action missing url")
                self._page.goto(url, timeout=self.navigation_timeout_ms)
            elif action_type in ("wait", "pause"):
                duration_ms = action.get("duration_ms", 1000)
                time.sleep(duration_ms / 1000.0)
            elif action_type in ("screenshot", ""):
                pass
            else:
                raise ValueError(f"Unsupported computer action: {action_type}")

            if self.wait_after_action_ms > 0:
                time.sleep(self.wait_after_action_ms / 1000.0)

        except Exception as exc:
            success = False
            error = str(exc)

        screenshot_bytes = b""
        try:
            screenshot_bytes = self._page.screenshot(
                type="png",
                full_page=self.full_page
            )
        except Exception as exc:
            if success:
                success = False
                error = f"Screenshot failed: {exc}"

        screenshot_b64 = base64.b64encode(screenshot_bytes).decode("utf-8") if screenshot_bytes else ""

        return {
            "success": success,
            "error": error,
            "action_type": action_type,
            "screenshot_b64": screenshot_b64,
            "screenshot_bytes": len(screenshot_bytes),
            "current_url": self._page.url if self._page else "",
        }


class ComputerUseScenario(BaseScenario):
    """
    Computer use scenario using the OpenAI computer tool.
    """

    @property
    def scenario_type(self) -> str:
        return "computer_use"

    def run(self, network_profile: str, run_index: int = 0) -> ScenarioResult:
        session_id = self._create_session_id()
        model = self.config.get("model", "computer-use-preview")
        prompts = self.config.get("prompts", [
            "Open https://example.com and summarize the page title."
        ])
        max_steps = int(self.config.get("max_steps", 15))

        backend = self.config.get("execution_backend", "playwright")
        viewport = self.config.get("viewport", {"width": 1280, "height": 720})
        full_page = bool(self.config.get("full_page_screenshot", True))
        navigation_timeout_ms = int(self.config.get("navigation_timeout_ms", 30000))
        wait_after_action_ms = int(self.config.get("wait_after_action_ms", 250))
        environment = self.config.get("environment", "browser")

        executor = ComputerUseExecutor(
            backend=backend,
            viewport=(int(viewport.get("width", 1280)), int(viewport.get("height", 720))),
            full_page=full_page,
            navigation_timeout_ms=navigation_timeout_ms,
            wait_after_action_ms=wait_after_action_ms,
        )

        result = ScenarioResult(
            scenario_id=self.scenario_id,
            session_id=session_id,
            network_profile=network_profile,
            run_index=run_index,
        )
        result.metadata.update({
            "computer_backend": backend,
            "viewport": executor.viewport,
            "full_page_screenshot": full_page,
        })

        self._openai = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

        try:
            executor.start()

            for turn_index, prompt in enumerate(prompts):
                response = None
                step_index = 0
                pending_input = self._build_user_input(prompt)

                while step_index < max_steps:
                    response, response_record = self._create_response_record(
                        pending_input,
                        model,
                        network_profile,
                        session_id,
                        turn_index,
                        run_index,
                        step_index,
                        previous_response_id=response.id if response else None,
                    )

                    self.logger.log(response_record)
                    result.log_records.append(response_record)

                    result.api_call_count += 1
                    result.total_latency_sec += response_record.latency_sec
                    result.total_request_bytes += response_record.request_bytes
                    result.total_response_bytes += response_record.response_bytes
                    if response_record.tokens_in:
                        result.total_tokens_in += response_record.tokens_in
                    if response_record.tokens_out:
                        result.total_tokens_out += response_record.tokens_out

                    tool_calls = self._extract_computer_calls(response)
                    if not tool_calls:
                        break

                    tool_outputs = []
                    for action_index, call in enumerate(tool_calls):
                        t_tool_start = time.time()
                        action_result = executor.execute(call.action)
                        tool_latency = time.time() - t_tool_start

                        tool_output = self._build_tool_output(call, action_result)
                        tool_outputs.append(tool_output)

                        action_request_bytes = len(
                            json.dumps(call.action).encode("utf-8")
                        )
                        screenshot_bytes = action_result.get("screenshot_bytes", 0)
                        screenshot_hash = ""
                        if action_result.get("screenshot_b64"):
                            screenshot_hash = hashlib.sha256(
                                action_result["screenshot_b64"].encode("utf-8")
                            ).hexdigest()

                        tool_record = self._create_log_record(
                            session_id=session_id,
                            turn_index=turn_index,
                            run_index=run_index,
                            network_profile=network_profile,
                            request_bytes=action_request_bytes,
                            response_bytes=screenshot_bytes,
                            t_request_start=t_tool_start,
                            latency_sec=tool_latency,
                            http_status=200 if action_result.get("success") else 500,
                            success=bool(action_result.get("success")),
                            error_type=action_result.get("error"),
                            tool_calls_count=1,
                            total_tool_bytes=action_request_bytes + screenshot_bytes,
                            tool_latency_sec=tool_latency,
                            metadata=json.dumps({
                                "type": "computer_use_action",
                                "call_id": call.call_id,
                                "action_type": action_result.get("action_type"),
                                "action_index": action_index,
                                "step_index": step_index,
                                "backend": backend,
                                "screenshot_bytes": screenshot_bytes,
                                "screenshot_hash": screenshot_hash,
                                "current_url": action_result.get("current_url"),
                                "action_summary": self._summarize_action(call.action),
                            })
                        )
                        self.logger.log(tool_record)
                        result.log_records.append(tool_record)

                        result.tool_calls_count += 1
                        result.tool_total_latency_sec += tool_latency
                        result.total_request_bytes += action_request_bytes
                        result.total_response_bytes += screenshot_bytes

                    pending_input = tool_outputs
                    step_index += 1

                result.turn_count += 1

            if response:
                result.metadata["final_response_text"] = self._extract_output_text(response)

        except Exception as exc:
            result.success = False
            result.error_message = str(exc)
        finally:
            executor.close()

        return result

    def _build_user_input(self, prompt: str) -> list[dict]:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt}
                ]
            }
        ]

    def _build_tool_output(self, call: ComputerToolCall, action_result: dict) -> dict:
        screenshot_b64 = action_result.get("screenshot_b64", "")
        output_payload = {
            "type": "computer_screenshot",
            "image_url": f"data:image/png;base64,{screenshot_b64}" if screenshot_b64 else "",
        }
        output_type = "computer_call_output" if call.call_type == "computer_call" else "tool_call_output"
        return {
            "type": output_type,
            "call_id": call.call_id,
            "output": output_payload,
        }

    def _create_response_record(
        self,
        input_items: list[dict],
        model: str,
        network_profile: str,
        session_id: str,
        turn_index: int,
        run_index: int,
        step_index: int,
        previous_response_id: Optional[str] = None,
    ) -> tuple[Any, LogRecord]:
        tool_payload = {
            "type": "computer_use_preview",
            "display_width": int(self.config.get("viewport", {}).get("width", 1024)),
            "display_height": int(self.config.get("viewport", {}).get("height", 768)),
            "environment": self.config.get("environment", "browser"),
        }
        payload: dict[str, Any] = {
            "model": model,
            "tools": [tool_payload],
            "input": input_items,
            "truncation": "auto",
        }
        if previous_response_id:
            payload["previous_response_id"] = previous_response_id

        request_bytes = len(json.dumps(payload).encode("utf-8"))
        t_request_start = time.time()

        response = self._openai.responses.create(**payload)
        latency_sec = time.time() - t_request_start

        response_dict = self._as_dict(response)
        response_bytes = len(json.dumps(response_dict).encode("utf-8"))

        tokens_in, tokens_out = self._extract_usage_tokens(response)
        if tokens_in is None:
            try:
                tokens_in = self.client.estimate_tokens(json.dumps(input_items), model)
            except Exception:
                tokens_in = None
        if tokens_out is None:
            output_text = self._extract_output_text(response)
            if output_text:
                tokens_out = self.client.estimate_tokens(output_text, model)
        tool_calls = self._extract_computer_calls(response)

        model_alias = self._anonymizer.model_alias(model)
        record = self._create_log_record(
            session_id=session_id,
            turn_index=turn_index,
            run_index=run_index,
            network_profile=network_profile,
            request_bytes=request_bytes,
            response_bytes=response_bytes,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            t_request_start=t_request_start,
            latency_sec=latency_sec,
            http_status=200,
            success=True,
            tool_calls_count=len(tool_calls),
            trace_request={
                "format": "openai.responses.create",
                "payload": payload,
            },
            trace_response={
                "format": "openai.responses.response",
                "payload": response_dict,
            },
            trace_note="computer_use_step",
            metadata=json.dumps({
                "type": "computer_use_response",
                "response_id": getattr(response, "id", None),
                "status": getattr(response, "status", None),
                "step_index": step_index,
                "tool_calls": len(tool_calls),
                "model": model_alias,
            })
        )

        return response, record

    def _extract_computer_calls(self, response: Any) -> list[ComputerToolCall]:
        output_items = []
        if hasattr(response, "output"):
            output_items = response.output or []
        elif isinstance(response, dict):
            output_items = response.get("output", [])

        calls: list[ComputerToolCall] = []

        for item in output_items:
            item_dict = self._as_dict(item)
            item_type = (item_dict.get("type") or "").lower()
            if item_type not in ("computer_call", "tool_call"):
                continue

            name = item_dict.get("name") or item_dict.get("tool_name")
            if item_type == "tool_call" and name not in ("computer", "computer_use", "computer_use_preview"):
                continue

            call_id = item_dict.get("call_id") or item_dict.get("id") or ""
            action = item_dict.get("action")

            if action is None:
                arguments = item_dict.get("arguments") or {}
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {}
                action = arguments.get("action") if isinstance(arguments, dict) else None
                if action is None and isinstance(arguments, dict):
                    action = arguments

            if not isinstance(action, dict):
                continue

            calls.append(ComputerToolCall(
                call_id=call_id,
                action=action,
                call_type=item_type,
            ))

        required_action = getattr(response, "required_action", None)
        if required_action:
            required_dict = self._as_dict(required_action)
            tool_calls = required_dict.get("tool_calls", [])
            for call in tool_calls:
                call_dict = self._as_dict(call)
                name = call_dict.get("name")
                if name not in ("computer", "computer_use", "computer_use_preview"):
                    continue
                call_id = call_dict.get("call_id") or call_dict.get("id") or ""
                arguments = call_dict.get("arguments", {})
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {}
                action = arguments.get("action") if isinstance(arguments, dict) else None
                if action is None and isinstance(arguments, dict):
                    action = arguments
                if isinstance(action, dict):
                    calls.append(ComputerToolCall(
                        call_id=call_id,
                        action=action,
                        call_type="tool_call",
                    ))

        return calls

    def _extract_output_text(self, response: Any) -> str:
        if hasattr(response, "output_text") and response.output_text:
            return response.output_text

        output_items = []
        if hasattr(response, "output"):
            output_items = response.output or []
        elif isinstance(response, dict):
            output_items = response.get("output", [])

        texts = []
        for item in output_items:
            item_dict = self._as_dict(item)
            if (item_dict.get("type") or "").lower() != "message":
                continue
            for content in item_dict.get("content", []):
                if content.get("type") == "text":
                    texts.append(content.get("text", ""))

        return "\n".join(texts)

    def _extract_usage_tokens(self, response: Any) -> tuple[Optional[int], Optional[int]]:
        usage = getattr(response, "usage", None)
        if not usage:
            return None, None

        usage_dict = self._as_dict(usage)
        tokens_in = usage_dict.get("input_tokens") or usage_dict.get("prompt_tokens")
        tokens_out = usage_dict.get("output_tokens") or usage_dict.get("completion_tokens")
        return tokens_in, tokens_out

    def _as_dict(self, obj: Any) -> dict:
        if isinstance(obj, dict):
            return obj
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        if hasattr(obj, "dict"):
            return obj.dict()
        return obj.__dict__ if hasattr(obj, "__dict__") else {}

    def _summarize_action(self, action: dict) -> dict:
        summary = dict(action)
        text = summary.pop("text", None)
        if text is not None:
            summary["text_len"] = len(text)
            summary["text_preview"] = text[:80]
        return summary
