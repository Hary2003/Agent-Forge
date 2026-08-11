"""
A small set of built-in tools so workflows are useful out of the box.
Add your own with the same @tool pattern in a separate module and import
it before running a workflow so the registry picks it up.
"""
from __future__ import annotations

import io
import contextlib
import os

from agentforge.tools.registry import tool


def _output_dir() -> str:
    return os.environ.get("AGENTFORGE_OUTPUT_DIR", "./outputs")


@tool("write_file", "Write text content to a file under the workflow's output directory.")
def write_file(filename: str, content: str) -> str:
    output_dir = _output_dir()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Wrote {len(content)} chars to {path}"


@tool("read_file", "Read text content from a file.")
def read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


@tool(
    "run_python",
    "Execute a short Python snippet for data analysis or chart generation. "
    "stdout is captured and returned. Use write_file separately to persist outputs.",
)
def run_python(code: str) -> str:
    buf = io.StringIO()
    local_vars: dict = {}
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, {"__builtins__": __builtins__}, local_vars)
    except Exception as e:  # noqa: BLE001 - deliberately broad, surfaced to the agent
        return f"ERROR: {e}\n---stdout so far---\n{buf.getvalue()}"
    return buf.getvalue() or "(no stdout)"
