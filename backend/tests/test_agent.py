import sys

sys.path.insert(0, "src")

import pytest

import agentforge.tools.builtin  # noqa: F401  (registers built-in tools)
from agentforge.core.agent import Agent
from agentforge.core.llm import LLMResponse, LLMToolUse, MockLLMClient
from agentforge.core.schema import AgentNodeConfig, NodeStatus


@pytest.mark.asyncio
async def test_required_write_file_falls_back_when_model_never_calls_tool(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTFORGE_OUTPUT_DIR", str(tmp_path))
    llm = MockLLMClient(canned_text="final report")
    agent = Agent(
        AgentNodeConfig(
            id="writer",
            role="writer",
            goal="write and save a report",
            tools=["write_file"],
            required_tools=["write_file"],
        ),
        llm,
    )

    result = await agent.run("make a report", {})

    assert result.status == NodeStatus.SUCCESS
    assert result.text == "final report"
    assert len(result.tool_calls) == 1
    call = result.tool_calls[0]
    assert call.name == "write_file"
    assert call.error is None
    assert call.arguments["_note"] == "auto-fallback: framework invoked this automatically"
    assert (tmp_path / "writer.md").read_text(encoding="utf-8") == "final report"


@pytest.mark.asyncio
async def test_required_write_file_no_fallback_when_model_calls_tool_itself(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTFORGE_OUTPUT_DIR", str(tmp_path))

    class ToolCallingLLM(MockLLMClient):
        async def complete(self, *, system, messages, tools=None, model=None, max_tokens=4096):
            self.calls.append({"system": system, "messages": messages, "tools": tools})
            if len(self.calls) == 1:
                return LLMResponse(
                    text="",
                    tool_uses=[
                        LLMToolUse(
                            id="toolu_1",
                            name="write_file",
                            arguments={"filename": "manual.md", "content": "saved by model"},
                        )
                    ],
                    stop_reason="tool_use",
                )
            return LLMResponse(text="done", tool_uses=[], stop_reason="end_turn")

    llm = ToolCallingLLM()
    agent = Agent(
        AgentNodeConfig(
            id="writer",
            role="writer",
            goal="write and save a report",
            tools=["write_file"],
            required_tools=["write_file"],
        ),
        llm,
    )

    result = await agent.run("make a report", {})

    assert result.status == NodeStatus.SUCCESS
    assert len(result.tool_calls) == 1
    call = result.tool_calls[0]
    assert call.name == "write_file"
    assert call.error is None
    assert "_note" not in call.arguments
    assert (tmp_path / "manual.md").read_text(encoding="utf-8") == "saved by model"
    assert not any(
        "Before finishing, you still need to call" in str(message["content"])
        for call_record in llm.calls
        for message in call_record["messages"]
    )
