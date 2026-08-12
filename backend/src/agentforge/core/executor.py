"""
DAG executor.

Given a WorkflowConfig, builds a dependency graph from each node's
`inputs`, validates it (no cycles, no missing references), then executes
nodes in topological order — running all nodes whose dependencies are
satisfied concurrently via asyncio.gather, rather than one-at-a-time like
a simple chain.
"""
from __future__ import annotations

import asyncio
from typing import Optional

from agentforge.core.agent import Agent
from agentforge.core.llm import LLMClient
from agentforge.core.schema import (
    AgentNodeConfig,
    NodeResult,
    NodeStatus,
    WorkflowConfig,
    WorkflowRunResult,
)
from agentforge.tools.registry import ToolRegistry, default_registry


class GraphError(ValueError):
    pass


def _topological_batches(nodes: list[AgentNodeConfig]) -> list[list[AgentNodeConfig]]:
    """Groups nodes into ordered batches; each batch can run in parallel."""
    by_id = {n.id: n for n in nodes}
    for n in nodes:
        for dep in n.inputs:
            if dep not in by_id:
                raise GraphError(f"Node '{n.id}' depends on unknown node '{dep}'")

    remaining = {n.id: set(n.inputs) for n in nodes}
    batches: list[list[AgentNodeConfig]] = []

    while remaining:
        ready = [nid for nid, deps in remaining.items() if not deps]
        if not ready:
            raise GraphError(f"Cycle detected among nodes: {list(remaining)}")
        batches.append([by_id[nid] for nid in ready])
        for nid in ready:
            del remaining[nid]
        for deps in remaining.values():
            deps.difference_update(ready)

    return batches


class WorkflowExecutor:
    def __init__(self, llm: LLMClient, registry: Optional[ToolRegistry] = None):
        self.llm = llm
        self.registry = registry or default_registry

    async def run(
        self,
        workflow: WorkflowConfig,
        task_prompt: str,
        on_node_complete=None,
        on_node_start=None,
        on_tool_call=None,
    ) -> WorkflowRunResult:
        """
        Executes `workflow` against `task_prompt`.

        on_node_complete: optional callback(node_id, NodeResult)
        on_node_start: optional callback(node_id)
        on_tool_call: optional callback(node_id, ToolCall)
        """
        batches = _topological_batches(workflow.nodes)
        run_result = WorkflowRunResult(workflow_name=workflow.name, task_prompt=task_prompt)

        async def _call_cb(cb, *args):
            if cb:
                res = cb(*args)
                if asyncio.iscoroutine(res):
                    await res

        for batch in batches:
            tasks = []
            for node_cfg in batch:
                upstream = {
                    dep_id: run_result.node_results[dep_id]
                    for dep_id in node_cfg.inputs
                }
                tasks.append(
                    self._run_node(
                        node_cfg,
                        task_prompt,
                        upstream,
                        on_node_start=on_node_start,
                        on_tool_call=on_tool_call,
                    )
                )

            results = await asyncio.gather(*tasks)

            for node_cfg, result in zip(batch, results):
                run_result.node_results[node_cfg.id] = result
                if on_node_complete:
                    await _call_cb(on_node_complete, node_cfg.id, result)
                if result.status == NodeStatus.FAILED:
                    # Downstream nodes that depend on a failed node are marked
                    # skipped rather than silently run on missing context.
                    self._skip_downstream(workflow, node_cfg.id, run_result)

        # Convention: the terminal node (no other node depends on it) is the
        # workflow's final output. If there are multiple, the last-defined wins.
        run_result.final_node_id = _find_terminal_node(workflow)
        return run_result

    async def _run_node(
        self,
        node_cfg: AgentNodeConfig,
        task_prompt: str,
        upstream: dict[str, NodeResult],
        on_node_start=None,
        on_tool_call=None,
    ) -> NodeResult:
        async def _call_cb(cb, *args):
            if cb:
                res = cb(*args)
                if asyncio.iscoroutine(res):
                    await res

        if any(r.status in (NodeStatus.FAILED, NodeStatus.SKIPPED) for r in upstream.values()):
            return NodeResult(
                node_id=node_cfg.id,
                status=NodeStatus.SKIPPED,
                error="Skipped because an upstream dependency failed.",
            )
        await _call_cb(on_node_start, node_cfg.id)
        from agentforge.core.node import create_node
        node = create_node(node_cfg, self.llm, self.registry)
        try:
            return await node.run(task_prompt, upstream, on_tool_call=on_tool_call)
        except Exception as exc:  # noqa: BLE001 - keep failures in the workflow result
            return NodeResult(
                node_id=node_cfg.id,
                status=NodeStatus.FAILED,
                error=str(exc),
            )

    def _skip_downstream(self, workflow: WorkflowConfig, failed_id: str, run_result: WorkflowRunResult):
        for n in workflow.nodes:
            if failed_id in n.inputs and n.id not in run_result.node_results:
                run_result.node_results[n.id] = NodeResult(
                    node_id=n.id, status=NodeStatus.SKIPPED,
                    error=f"Skipped: depends on failed node '{failed_id}'",
                )


def _find_terminal_node(workflow: WorkflowConfig) -> Optional[str]:
    depended_on = {dep for n in workflow.nodes for dep in n.inputs}
    terminals = [n.id for n in workflow.nodes if n.id not in depended_on]
    return terminals[-1] if terminals else (workflow.nodes[-1].id if workflow.nodes else None)
