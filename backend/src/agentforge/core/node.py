"""
BaseNode abstract interface for AgentForge DAG nodes.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

from agentforge.core.schema import NodeResult, ToolCall


class BaseNode(ABC):
    """Abstract interface that all node types in AgentForge must implement."""

    @abstractmethod
    async def run(
        self,
        task_prompt: str,
        upstream: dict[str, NodeResult],
        on_tool_call: Optional[Callable[[str, ToolCall], Any]] = None,
    ) -> NodeResult:
        """
        Execute the node logic.

        :param task_prompt: The global task prompt for the workflow run.
        :param upstream: Dictionary mapping upstream node IDs to their NodeResult objects.
        :param on_tool_call: Optional callback for intermediate events/tool calls.
        :return: NodeResult containing output text, artifacts, structured data, or errors.
        """
        pass


def create_node(
    config: "AgentNodeConfig",
    llm: Any,
    registry: Any = None,
) -> BaseNode:
    """Node factory creating the appropriate BaseNode implementation based on node_type."""
    ntype = (config.node_type or "agent").lower()
    if ntype in ("retrieval", "retriever", "rag"):
        from agentforge.rag.retrieval_node import RetrievalNode
        return RetrievalNode(config)
    else:
        from agentforge.core.agent import Agent
        return Agent(config, llm=llm, registry=registry)

