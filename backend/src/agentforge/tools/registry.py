"""
Tool registry.

Tools are plain Python callables decorated with `@tool`. The decorator
captures a name, description, and a JSON schema for arguments (either
hand-written or inferred from type hints), so the same function can be:
  1. exposed to the LLM as a callable tool, and
  2. invoked directly by the executor when the model asks for it.
"""
from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass
from typing import Any, Callable, Optional, get_type_hints


_PY_TO_JSON_TYPE = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


@dataclass
class RegisteredTool:
    name: str
    description: str
    func: Callable
    schema: dict[str, Any]

    def to_anthropic_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.schema,
        }

    async def invoke(self, **kwargs) -> Any:
        if inspect.iscoroutinefunction(self.func):
            return await self.func(**kwargs)
        return await asyncio.to_thread(self.func, **kwargs)


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, RegisteredTool] = {}

    def register(self, name: str, description: str, func: Callable) -> None:
        schema = _infer_schema(func)
        self._tools[name] = RegisteredTool(name=name, description=description, func=func, schema=schema)

    def get(self, name: str) -> RegisteredTool:
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' is not registered. Known tools: {list(self._tools)}")
        return self._tools[name]

    def subset(self, names: list[str]) -> list[RegisteredTool]:
        return [self.get(n) for n in names]

    def all(self) -> list[RegisteredTool]:
        return list(self._tools.values())

    def list_tools(self) -> list[RegisteredTool]:
        return self.all()


def _infer_schema(func: Callable) -> dict[str, Any]:
    sig = inspect.signature(func)
    hints = get_type_hints(func)
    properties: dict[str, Any] = {}
    required: list[str] = []
    for pname, param in sig.parameters.items():
        if pname == "self":
            continue
        py_type = hints.get(pname, str)
        json_type = _PY_TO_JSON_TYPE.get(py_type, "string")
        properties[pname] = {
            "type": json_type,
            "description": f"The {pname} parameter for {func.__name__}.",
        }
        if param.default is inspect.Parameter.empty:
            required.append(pname)
    return {"type": "object", "properties": properties, "required": required}


# Global default registry. Modules can import and add to this, or build
# their own ToolRegistry() instance for isolated test setups.
default_registry = ToolRegistry()


def tool(name: Optional[str] = None, description: str = "", registry: Optional[ToolRegistry] = None):
    """Decorator: @tool("web_search", "Search the web") def web_search(query: str) -> str: ..."""
    def decorator(func: Callable) -> Callable:
        target_registry = registry or default_registry
        target_registry.register(name or func.__name__, description or (func.__doc__ or ""), func)
        return func
    return decorator
