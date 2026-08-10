#!/usr/bin/env python3
"""CLI entry point for running an agentforge workflow."""
from __future__ import annotations

import argparse
import asyncio
import sys

sys.path.insert(0, "src")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import agentforge.tools.builtin  # noqa: E402,F401  (registers built-in tools)
from agentforge.core.executor import WorkflowExecutor  # noqa: E402
from agentforge.core.llm import GroqClient, MockLLMClient  # noqa: E402
from agentforge.core.schema import NodeStatus  # noqa: E402
from agentforge.core.workflow import load_workflow  # noqa: E402


def _on_node_complete(node_id: str, result) -> None:
    label = {"success": "OK", "failed": "FAIL", "skipped": "SKIP"}.get(
        result.status.value, "?"
    )
    print(f"[{label}] {node_id} ({result.status.value})")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run an agentforge workflow.")
    parser.add_argument("--workflow", required=True, help="Path to a workflow YAML file")
    parser.add_argument("--task", required=True, help="Task prompt for the workflow")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use MockLLMClient (no API key or network needed)",
    )
    parser.add_argument(
        "--model",
        default="llama-3.3-70b-versatile",
        help="Default model for nodes that don't override it",
    )
    args = parser.parse_args()

    workflow = load_workflow(args.workflow)
    llm = MockLLMClient() if args.mock else GroqClient(default_model=args.model)
    executor = WorkflowExecutor(llm=llm)

    print(f"Running workflow '{workflow.name}': {workflow.description.strip()}\n")
    result = await executor.run(workflow, args.task, on_node_complete=_on_node_complete)

    print("\n--- Node outputs ---")
    for node_id, node_result in result.node_results.items():
        print(f"\n## {node_id} [{node_result.status.value}]")
        if node_result.status == NodeStatus.SUCCESS:
            print(node_result.text)
        elif node_result.error:
            print(f"  {node_result.error}")

    if result.final_result:
        print("\n=== Final output ===")
        print(result.final_result.text)


if __name__ == "__main__":
    asyncio.run(main())
