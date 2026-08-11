import sys
sys.path.insert(0, "src")

import pytest

from agentforge.core.executor import WorkflowExecutor, GraphError, _topological_batches
from agentforge.core.llm import MockLLMClient
from agentforge.core.schema import AgentNodeConfig, NodeStatus, WorkflowConfig


def make_workflow():
    return WorkflowConfig(
        name="test_wf",
        nodes=[
            AgentNodeConfig(id="a", role="r1", goal="do a", inputs=[]),
            AgentNodeConfig(id="b", role="r2", goal="do b", inputs=[]),
            AgentNodeConfig(id="c", role="r3", goal="merge", inputs=["a", "b"]),
        ],
    )


def test_topological_batches_parallelizes_independent_nodes():
    wf = make_workflow()
    batches = _topological_batches(wf.nodes)
    # a and b have no deps -> same batch; c depends on both -> its own batch after
    assert {n.id for n in batches[0]} == {"a", "b"}
    assert [n.id for n in batches[1]] == ["c"]


def test_cycle_detection():
    wf = WorkflowConfig(
        name="cyclic",
        nodes=[
            AgentNodeConfig(id="a", role="r", goal="g", inputs=["b"]),
            AgentNodeConfig(id="b", role="r", goal="g", inputs=["a"]),
        ],
    )
    with pytest.raises(GraphError):
        _topological_batches(wf.nodes)


def test_missing_dependency_raises():
    wf = WorkflowConfig(
        name="bad",
        nodes=[AgentNodeConfig(id="a", role="r", goal="g", inputs=["ghost"])],
    )
    with pytest.raises(GraphError):
        _topological_batches(wf.nodes)


@pytest.mark.asyncio
async def test_executor_runs_and_merges_upstream_context():
    wf = make_workflow()
    llm = MockLLMClient(canned_text="ok")
    executor = WorkflowExecutor(llm=llm)
    result = await executor.run(wf, "test task")

    assert result.node_results["a"].status == NodeStatus.SUCCESS
    assert result.node_results["b"].status == NodeStatus.SUCCESS
    assert result.node_results["c"].status == NodeStatus.SUCCESS
    assert result.final_node_id == "c"

    # The merge node's prompt should have included both upstream outputs.
    c_call_messages = llm.calls[-1]["messages"]
    user_prompt = c_call_messages[0]["content"]
    assert "Output of 'a'" in user_prompt
    assert "Output of 'b'" in user_prompt


@pytest.mark.asyncio
async def test_downstream_skipped_when_upstream_fails():
    wf = make_workflow()

    class FailingLLM(MockLLMClient):
        async def complete(self, *, system, messages, tools=None, model=None, max_tokens=4096):
            if "do a" in system:
                raise RuntimeError("simulated failure")
            return await super().complete(system=system, messages=messages, tools=tools, model=model)

    executor = WorkflowExecutor(llm=FailingLLM())
    result = await executor.run(wf, "test task")

    assert result.node_results["a"].status == NodeStatus.FAILED
    assert result.node_results["b"].status == NodeStatus.SUCCESS
    assert result.node_results["c"].status == NodeStatus.SKIPPED
    assert "simulated failure" in result.node_results["a"].error
