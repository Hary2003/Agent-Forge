"""Thin LLM client abstraction and Groq adapter."""
from __future__ import annotations

import asyncio
import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class LLMToolUse:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    text: str = ""
    tool_uses: list[LLMToolUse] = field(default_factory=list)
    stop_reason: str = "end_turn"
    raw: Any = None


class LLMClient(ABC):
    @abstractmethod
    async def complete(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        model: Optional[str] = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        ...


class GroqClient(LLMClient):
    """
    Groq SDK adapter.

    The framework uses an Anthropic-style internal format:
    content blocks with text/tool_use/tool_result. Groq's API is
    OpenAI-compatible, so this client translates tools and messages at
    the boundary.
    """

    def __init__(self, default_model: str = "llama-3.3-70b-versatile"):
        self.default_model = default_model
        self._client = None
        self._semaphore = None

    def _get_client(self):
        if self._client is None:
            from groq import AsyncGroq

            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                try:
                    from dotenv import load_dotenv
                    load_dotenv()
                    api_key = os.environ.get("GROQ_API_KEY")
                except ImportError:
                    pass
            if not api_key:
                raise RuntimeError(
                    "GROQ_API_KEY is not set. Get a key at https://console.groq.com/keys "
                    "and add it to your environment."
                )
            self._client = AsyncGroq(api_key=api_key, timeout=30.0, max_retries=3)
        return self._client

    async def complete(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        model: Optional[str] = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(2)

        async with self._semaphore:
            kwargs: dict[str, Any] = {
                "model": model or self.default_model,
                "max_tokens": max_tokens,
                "messages": _to_openai_messages(system, messages),
            }
            if tools:
                kwargs["tools"] = [_to_openai_tool_schema(t) for t in tools]
                kwargs["tool_choice"] = "auto"

            max_attempts = 3
            last_err = None
            for attempt in range(max_attempts):
                try:
                    resp = await self._get_client().chat.completions.create(**kwargs)
                    choice = resp.choices[0]
                    msg = choice.message

                    tool_uses: list[LLMToolUse] = []
                    for call in msg.tool_calls or []:
                        raw_args = call.function.arguments or "{}"
                        try:
                            arguments = json.loads(raw_args)
                        except json.JSONDecodeError:
                            arguments = {"_raw_arguments": raw_args}
                        tool_uses.append(
                            LLMToolUse(id=call.id, name=call.function.name, arguments=arguments)
                        )

                    return LLMResponse(
                        text=msg.content or "",
                        tool_uses=tool_uses,
                        stop_reason="tool_use" if tool_uses else (choice.finish_reason or "end_turn"),
                        raw=resp,
                    )
                except Exception as err:
                    last_err = err
                    err_str = str(err)
                    if "failed_generation" in err_str or "tool_use_failed" in err_str:
                        # Attempt to extract function name and json arguments from Groq's failed generation output
                        match = re.search(
                            r"<function=([a-zA-Z0-9_]+)(?:\[.*?\])?\s*(\{[\s\S]*?\})\s*(?:</function>)?",
                            err_str,
                        )
                        if match:
                            fn_name = match.group(1)
                            fn_args_raw = match.group(2)
                            try:
                                arguments = json.loads(fn_args_raw)
                            except json.JSONDecodeError:
                                arguments = {"_raw": fn_args_raw}
                            return LLMResponse(
                                text="",
                                tool_uses=[
                                    LLMToolUse(
                                        id="call_fallback_groq",
                                        name=fn_name,
                                        arguments=arguments,
                                    )
                                ],
                                stop_reason="tool_use",
                            )

                    is_conn_err = any(
                        k in err_str.lower()
                        for k in ("connection", "connect", "timeout", "rate limit", "503", "502", "504")
                    )
                    if is_conn_err and attempt < max_attempts - 1:
                        await asyncio.sleep(1.5 * (attempt + 1))
                        continue
                    raise err


class MockLLMClient(LLMClient):
    """
    Deterministic offline client for testing graph logic without network access.
    Returns realistic node summaries and tool calls.
    """

    def __init__(self, canned_text: Optional[str] = None):
        self.canned_text = canned_text
        self.calls: list[dict[str, Any]] = []

    async def complete(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        model: Optional[str] = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        self.calls.append({"system": system, "messages": messages, "tools": tools})

        if self.canned_text:
            return LLMResponse(text=self.canned_text, tool_uses=[], stop_reason="end_turn")

        system_lower = system.lower()
        tool_names = [t["name"] for t in (tools or [])]

        # Check if write_file is requested and hasn't been called in this turn yet
        has_tool_results = any(
            isinstance(m.get("content"), list)
            and any(b.get("type") == "tool_result" for b in m["content"])
            for m in messages
        )

        if "write_file" in tool_names and not has_tool_results:
            # Determine suitable report text from context
            user_prompts = [
                m["content"]
                for m in messages
                if m.get("role") == "user" and isinstance(m.get("content"), str)
            ]
            task_context = "\n".join(user_prompts)

            report_body = (
                "# Comprehensive Agent-Forge Analysis Report\n\n"
                "## Executive Summary\n"
                "This report combines detailed research findings and quantitative takeaways from the multi-agent workflow.\n\n"
                "## 1. Research Findings\n"
                "Key technological milestones and architecture developments have been evaluated across industry standards.\n"
                "- High efficiency and parallel node execution achieved.\n"
                "- Modular tool integration enables flexible agent capabilities.\n\n"
                "## 2. Quantitative Takeaways\n"
                "- Workflow throughput: 100% completion across all dependency paths.\n"
                "- Latency optimization: Concurrent DAG execution reduces execution time by 45%.\n\n"
                "## Conclusion\n"
                "The multi-agent workflow architecture operates robustly with fault isolation and fallback support."
            )
            return LLMResponse(
                text="Generating final report...",
                tool_uses=[
                    LLMToolUse(
                        id="call_mock_write",
                        name="write_file",
                        arguments={"filename": "report.md", "content": report_body},
                    )
                ],
                stop_reason="tool_use",
            )

        if "researcher" in system_lower:
            text = (
                "### Research Findings\n"
                "1. Multi-agent DAG workflows enable scalable execution and distinct role isolation.\n"
                "2. Standardized tool definitions provide deterministic function execution.\n"
                "3. Key benchmarks demonstrate low latency and reliable agent output validation."
            )
        elif "data_analyst" in system_lower or "analyst" in system_lower:
            text = (
                "### Quantitative Analysis\n"
                "- Data metrics processed: 100% valid records.\n"
                "- Execution performance: 45% speed improvement via parallel DAG batches.\n"
                "- Success rate: All node dependencies satisfied cleanly."
            )
        elif "writer" in system_lower:
            text = "The comprehensive research and data analysis report has been compiled and saved to report.md."
        else:
            last_user = next(
                (m["content"] for m in reversed(messages) if m.get("role") == "user"), ""
            )
            text = f"[Mock response for task: {str(last_user)[:80]}]"

        return LLMResponse(text=text, tool_uses=[], stop_reason="end_turn")


def _to_openai_tool_schema(tool_schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool_schema["name"],
            "description": tool_schema.get("description", ""),
            "parameters": tool_schema.get("input_schema", {"type": "object"}),
        },
    }


def _to_openai_messages(system: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for message in messages:
        role = message["role"]
        content = message.get("content", "")

        if isinstance(content, str):
            converted.append({"role": role, "content": content})
            continue

        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for block in content:
            block_type = block.get("type")
            if block_type == "text":
                text_parts.append(block.get("text", ""))
            elif block_type == "tool_use":
                tool_calls.append(
                    {
                        "id": block["id"],
                        "type": "function",
                        "function": {
                            "name": block["name"],
                            "arguments": json.dumps(block.get("input", {})),
                        },
                    }
                )
            elif block_type == "tool_result":
                converted.append(
                    {
                        "role": "tool",
                        "tool_call_id": block["tool_use_id"],
                        "content": str(block.get("content", "")),
                    }
                )

        if tool_calls:
            converted.append(
                {
                    "role": "assistant",
                    "content": "\n".join(text_parts) if text_parts else None,
                    "tool_calls": tool_calls,
                }
            )
        elif text_parts:
            converted.append({"role": role, "content": "\n".join(text_parts)})

    return converted

