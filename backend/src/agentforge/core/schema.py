"""
Core data types shared across the framework.

Design notes:
- A Workflow is a DAG of AgentNodes.
- Each AgentNode declares which other nodes' outputs it depends on
  (`inputs`), so the executor can topologically sort and parallelize.
- NodeResult is the atomic unit passed between agents. Keeping it a
  plain, serializable object (not a raw string) means agents can pass
  structured data (file paths, tables, citations) rather than just text.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class NodeStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class ToolCall(BaseModel):
    """A single tool invocation an agent made, kept for tracing/debugging."""
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: Any = None
    error: Optional[str] = None


class NodeResult(BaseModel):
    """What an agent node produces. Passed downstream as input to dependents."""
    node_id: str
    status: NodeStatus = NodeStatus.PENDING
    text: str = ""                                  # primary natural-language output
    artifacts: dict[str, str] = Field(default_factory=dict)   # name -> filepath
    data: dict[str, Any] = Field(default_factory=dict)        # structured payload
    tool_calls: list[ToolCall] = Field(default_factory=list)
    error: Optional[str] = None

    def as_prompt_context(self) -> str:
        """Renders this result as text so it can be fed into a downstream agent's prompt."""
        parts = [self.text.strip()] if self.text.strip() else []
        if self.artifacts:
            parts.append("Artifacts produced: " + ", ".join(
                f"{name} -> {path}" for name, path in self.artifacts.items()
            ))
        return "\n".join(parts) if parts else "(no output)"


class AgentNodeConfig(BaseModel):
    """One node in the workflow graph, as authored in YAML."""
    id: str
    node_type: str = "agent"       # "agent", "retrieval", etc.
    role: str = ""                 # e.g. "researcher", "data_analyst", "writer"
    goal: str = ""                 # natural-language instructions for this agent
    inputs: list[str] = Field(default_factory=list)   # ids of upstream nodes
    tools: list[str] = Field(default_factory=list)    # names registered in tool registry
    required_tools: list[str] = Field(default_factory=list)  # tools that must be called >=1 time; enforced in code, see agent.py
    model: Optional[str] = None    # override default model for this node
    max_iterations: int = 6        # cap on tool-use loops before forcing a final answer
    config: dict[str, Any] = Field(default_factory=dict)  # node-specific settings (e.g. vector store, top_k, rerank)


class WorkflowConfig(BaseModel):
    """A full workflow: metadata + the DAG of nodes."""
    name: str
    description: str = ""
    nodes: list[AgentNodeConfig]

    def node_map(self) -> dict[str, AgentNodeConfig]:
        return {n.id: n for n in self.nodes}


class WorkflowRunResult(BaseModel):
    """Everything produced by one execution of a workflow."""
    workflow_name: str
    task_prompt: str
    node_results: dict[str, NodeResult] = Field(default_factory=dict)
    final_node_id: Optional[str] = None

    @property
    def final_result(self) -> Optional[NodeResult]:
        if self.final_node_id:
            return self.node_results.get(self.final_node_id)
        return None
