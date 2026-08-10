"""Thin LLM client abstraction and Groq adapter."""
from __future__ import annotations

import json
import os
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
            self._client = AsyncGroq(api_key=api_key)
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
        kwargs: dict[str, Any] = {
            "model": model or self.default_model,
            "max_tokens": max_tokens,
            "messages": _to_openai_messages(system, messages),
        }
        if tools:
            kwargs["tools"] = [_to_openai_tool_schema(t) for t in tools]
            kwargs["tool_choice"] = "auto"

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


class MockLLMClient(LLMClient):
    """
    Deterministic offline client for testing graph logic without network access.
    Returns canned text, no tool calls.
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
        last_user = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
        )
        text = self.canned_text or f"[mock response to: {str(last_user)[:80]}]"
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
                    "content": "\n".join(text_parts) or None,
                    "tool_calls": tool_calls,
                }
            )
        elif text_parts:
            converted.append({"role": role, "content": "\n".join(text_parts)})

    return converted
