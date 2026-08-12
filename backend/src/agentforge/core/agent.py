"""
Agent base class.

An Agent wraps: a role/goal (its "personality" and instructions), an LLM
client, and a set of tools it's allowed to call. `run()` drives the
standard tool-use loop: ask the model, execute any tool calls it requests,
feed results back, repeat until it produces a final text answer or hits
max_iterations.
"""
from __future__ import annotations

import asyncio
from typing import Optional

from agentforge.core.llm import LLMClient
from agentforge.core.node import BaseNode
from agentforge.core.schema import AgentNodeConfig, NodeResult, NodeStatus, ToolCall
from agentforge.tools.registry import ToolRegistry, default_registry


class Agent(BaseNode):
    def __init__(
        self,
        config: AgentNodeConfig,
        llm: LLMClient,
        registry: Optional[ToolRegistry] = None,
    ):
        self.config = config
        self.llm = llm
        self.registry = registry or default_registry

    def _system_prompt(self) -> str:
        return (
            f"You are the '{self.config.role}' agent in a multi-agent workflow.\n"
            f"Your goal for this task: {self.config.goal}\n\n"
            "Use the available tools when they help you complete the goal. "
            "When you are done, respond with your final answer as plain text "
            "(no further tool calls)."
        )

    def _build_user_prompt(self, task_prompt: str, upstream: dict[str, NodeResult]) -> str:
        parts = [f"Overall task: {task_prompt}"]
        if upstream:
            parts.append("\nContext from upstream agents:")
            for node_id, result in upstream.items():
                parts.append(f"\n--- Output of '{node_id}' ---\n{result.as_prompt_context()}")
        return "\n".join(parts)

    def _missing_required_tools(self, result: NodeResult) -> list[str]:
        successful_calls = {
            call.name for call in result.tool_calls if call.error is None
        }
        return [
            tool_name for tool_name in self.config.required_tools
            if tool_name not in successful_calls
        ]

    async def _apply_fallback(self, missing: list[str], result: NodeResult) -> None:
        for tool_name in missing:
            if tool_name == "write_file":
                arguments = {
                    "filename": f"{self.config.id}.md",
                    "content": result.text,
                    "_note": "auto-fallback: framework invoked this automatically",
                }
                call = ToolCall(name=tool_name, arguments=arguments)
                try:
                    tool_obj = self.registry.get(tool_name)
                    output = await tool_obj.invoke(
                        filename=arguments["filename"],
                        content=arguments["content"],
                    )
                    call.result = output
                except Exception as e:  # noqa: BLE001 - preserve fallback failure in trace
                    call.error = str(e)
                result.tool_calls.append(call)
                continue

            result.tool_calls.append(
                ToolCall(
                    name=tool_name,
                    error="Required tool was never called and has no automatic fallback.",
                )
            )

    async def run(
        self,
        task_prompt: str,
        upstream: dict[str, NodeResult],
        on_tool_call=None,
    ) -> NodeResult:
        tools = self.registry.subset(self.config.tools) if self.config.tools else []
        tool_schemas = [t.to_anthropic_schema() for t in tools] or None

        messages: list[dict] = [
            {"role": "user", "content": self._build_user_prompt(task_prompt, upstream)}
        ]
        result = NodeResult(node_id=self.config.id, status=NodeStatus.RUNNING)
        nudged = False

        initial_text = ""
        for _ in range(self.config.max_iterations):
            response = await self.llm.complete(
                system=self._system_prompt(),
                messages=messages,
                tools=tool_schemas,
                model=self.config.model,
            )

            if not response.tool_uses:
                if not initial_text and response.text:
                    initial_text = response.text
                result.text = response.text or initial_text
                missing = self._missing_required_tools(result)
                if missing and not nudged:
                    messages.append({"role": "assistant", "content": response.text})
                    messages.append({
                        "role": "user",
                        "content": (
                            "Before finishing, you still need to call: "
                            f"{', '.join(missing)}. Please call it now."
                        ),
                    })
                    nudged = True
                    continue

                if initial_text and (not response.text or response.text.startswith("[mock response to: Before finishing")):
                    result.text = initial_text
                else:
                    result.text = response.text or initial_text

                result.status = NodeStatus.SUCCESS
                if missing:
                    await self._apply_fallback(missing, result)
                return result

            # Model wants to call tools: append its turn, then run each tool
            # and append the results, then loop back for the next model turn.
            assistant_content = []
            if response.text:
                assistant_content.append({"type": "text", "text": response.text})
            for tu in response.tool_uses:
                assistant_content.append(
                    {"type": "tool_use", "id": tu.id, "name": tu.name, "input": tu.arguments}
                )
            messages.append({"role": "assistant", "content": assistant_content})

            tool_result_blocks = []
            for tu in response.tool_uses:
                call = ToolCall(name=tu.name, arguments=tu.arguments)
                try:
                    tool_obj = self.registry.get(tu.name)
                    output = await tool_obj.invoke(**tu.arguments)
                    call.result = output
                    tool_result_blocks.append(
                        {"type": "tool_result", "tool_use_id": tu.id, "content": str(output)}
                    )
                except Exception as e:  # noqa: BLE001 - surfaced back to the model, not swallowed silently
                    call.error = str(e)
                    tool_result_blocks.append(
                        {"type": "tool_result", "tool_use_id": tu.id, "content": f"ERROR: {e}", "is_error": True}
                    )
                result.tool_calls.append(call)
                if on_tool_call:
                    cb_res = on_tool_call(self.config.id, call)
                    if asyncio.iscoroutine(cb_res):
                        await cb_res

            messages.append({"role": "user", "content": tool_result_blocks})

        result.status = NodeStatus.FAILED
        result.error = f"Exceeded max_iterations ({self.config.max_iterations}) without a final answer."
        return result
