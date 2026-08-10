# agentforge

A lean, DAG-based multi-agent task orchestration framework — inspired by
[ChatDev/DevAll](https://github.com/OpenBMB/ChatDev), but scoped down to the
core pieces you actually need to get started: define agents in YAML, wire
them into a dependency graph, run it.

## Architecture

```
src/agentforge/
  core/
    schema.py     # Task/message/config data types (pydantic)
    llm.py        # Pluggable LLM client (Anthropic by default; Mock for offline testing)
    agent.py      # Agent base class: the tool-use loop
    executor.py   # DAG executor: topological batching + parallel execution
    workflow.py   # YAML -> WorkflowConfig loader
  tools/
    registry.py   # @tool decorator + JSON-schema inference
    builtin.py    # write_file, read_file, run_python
workflows/
  research_report.yaml   # example: two parallel agents merged by a writer
run.py            # CLI entry point
tests/
  test_executor.py        # DAG scheduling + failure-propagation tests
```

**How it works:** a workflow is a set of agent nodes, each declaring which
other nodes' outputs it depends on (`inputs`). The executor topologically
sorts the graph into batches — all nodes whose dependencies are satisfied
run concurrently (`asyncio.gather`), then the next batch runs, and so on.
This is the key upgrade over a simple linear chain: independent agents
(e.g. a researcher and a data analyst) run in parallel instead of waiting
on each other.

Each agent is just a role + goal + allowed tools. It runs a standard
tool-use loop against the LLM until it produces a final text answer or
hits `max_iterations`.

## Quick start

```bash
# 1. Install dependencies
pip install -e .

# 2. Configure your API key
cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY

# 3. Run the example workflow
python run.py --workflow workflows/research_report.yaml \
               --task "Research the state of solid-state batteries and summarize the market outlook"

# Or run offline with the mock LLM (no API key / network needed) to see the
# graph execution and tool-loop mechanics without spending tokens:
python run.py --workflow workflows/research_report.yaml --task "test" --mock
```

Outputs from `write_file` land in `./outputs/` by default (configurable via
`AGENTFORGE_OUTPUT_DIR`).

## Writing your own workflow

```yaml
name: my_workflow
nodes:
  - id: step_one
    role: researcher
    goal: "What this agent should accomplish, in plain language."
    inputs: []                # no dependencies -> runs first / in parallel
    tools: []                 # names must match a @tool-registered function

  - id: step_two
    role: writer
    goal: "Combine step_one's output into a final answer."
    inputs: [step_one]        # runs after step_one completes
    tools: [write_file]
```

## Adding a tool

```python
from agentforge.tools.registry import tool

@tool("web_search", "Search the web and return a summary of results.")
def web_search(query: str) -> str:
    ...  # your implementation
```

Import the module before running a workflow so the decorator registers it,
then reference the tool name in a node's `tools:` list.

## Running tests

```bash
pip install -e ".[dev]"
pytest
```

The test suite uses `MockLLMClient`, so it runs fully offline and verifies:
parallel batching of independent nodes, cycle detection, missing-dependency
validation, upstream-context passing, and failure propagation (a failed
node's dependents are marked `skipped` rather than run on incomplete data).

## Design choices vs. ChatDev

| | ChatDev 1.0 | ChatDev 2.0 (DevAll) | agentforge |
|---|---|---|---|
| Topology | Fixed chain | Configurable graph/orchestrator | DAG (parallel batches) |
| Config | Python | YAML + Vue web console | YAML |
| Scope | Software dev only | General ("develop everything") | General task orchestration |
| Complexity | Low | High (full platform: FastAPI + Vue + SDK) | Minimal — one Python package |

This is intentionally the "20% that gets you 80%": a real DAG scheduler and
tool-use loop, without ChatDev 2.0's full web console, database sync, or
orchestrator-agent complexity. Natural next additions once this is solid:
a web UI, a smarter dynamic orchestrator node, streaming progress over
websockets, and persistent run history.
