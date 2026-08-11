"""
FastAPI REST and WebSocket API server for Agent-Forge.
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

import agentforge.tools.builtin  # noqa: F401  Ensure tools are registered
from agentforge.core.executor import WorkflowExecutor
from agentforge.core.llm import GroqClient, MockLLMClient
from agentforge.core.schema import NodeResult, NodeStatus, ToolCall, WorkflowConfig
from agentforge.core.workflow import load_workflow
from agentforge.tools.registry import default_registry

app = FastAPI(
    title="Agent-Forge API",
    description="REST & WebSocket API for Agent-Forge DAG Workflow Orchestrator",
    version="0.1.0",
)

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.staticfiles import StaticFiles

# Base directories
BACKEND_DIR = Path(__file__).resolve().parents[3]
PROJECT_DIR = BACKEND_DIR.parent
WORKFLOWS_DIR = BACKEND_DIR / "workflows"
OUTPUTS_DIR = BACKEND_DIR / "outputs"
FRONTEND_DIST_DIR = PROJECT_DIR / "frontend" / "dist"

WORKFLOWS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


class WorkflowCreateRequest(BaseModel):
    name: str
    description: str = ""
    nodes: list[dict[str, Any]]


@app.get("/api/health")
def health_check():
    return {"status": "ok", "app": "agentforge"}


@app.get("/api/workflows")
def list_workflows():
    workflows = []
    if not WORKFLOWS_DIR.exists():
        return []
    for file in WORKFLOWS_DIR.glob("*.yaml"):
        try:
            wf = load_workflow(file)
            workflows.append(wf.model_dump())
        except Exception as e:
            workflows.append({
                "name": file.stem,
                "error": f"Failed to parse {file.name}: {e}",
                "description": "",
                "nodes": [],
            })
    return workflows


@app.get("/api/workflows/{name}")
def get_workflow(name: str):
    file_path = WORKFLOWS_DIR / f"{name}.yaml"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Workflow '{name}' not found")
    try:
        wf = load_workflow(file_path)
        return wf.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/workflows")
def create_or_update_workflow(req: WorkflowCreateRequest):
    import yaml

    file_path = WORKFLOWS_DIR / f"{req.name}.yaml"
    data = req.model_dump()
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, sort_keys=False)
        return {"status": "success", "name": req.name, "path": str(file_path)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/workflows/{name}")
def delete_workflow(name: str):
    file_path = WORKFLOWS_DIR / f"{name}.yaml"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Workflow '{name}' not found")
    file_path.unlink()
    return {"status": "deleted", "name": name}


@app.get("/api/tools")
def list_tools():
    tools = []
    for t in default_registry.all():
        tools.append({
            "name": t.name,
            "description": t.description,
            "schema": t.to_anthropic_schema(),
        })
    return tools


@app.get("/api/outputs")
def list_outputs():
    outputs = []
    if not OUTPUTS_DIR.exists():
        return []
    for file in OUTPUTS_DIR.glob("*"):
        if file.is_file():
            outputs.append({
                "name": file.name,
                "size": file.stat().st_size,
                "modified": file.stat().st_mtime,
                "path": str(file),
            })
    outputs.sort(key=lambda x: x["modified"], reverse=True)
    return outputs


@app.get("/api/outputs/{filename}")
def get_output_content(filename: str):
    file_path = OUTPUTS_DIR / filename
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found")
    try:
        content = file_path.read_text(encoding="utf-8")
        return PlainTextResponse(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/api/ws/run")
async def websocket_run(websocket: WebSocket):
    await websocket.accept()
    try:
        data_text = await websocket.receive_text()
        req = json.loads(data_text)

        workflow_name = req.get("workflow_name", "research_report")
        task_prompt = req.get("task_prompt", "Test task prompt")
        mock_mode = req.get("mock", True)
        model = req.get("model", "llama-3.3-70b-versatile")

        file_path = WORKFLOWS_DIR / f"{workflow_name}.yaml"
        if not file_path.exists():
            await websocket.send_json({
                "type": "error",
                "message": f"Workflow '{workflow_name}' file not found.",
            })
            await websocket.close()
            return

        workflow = load_workflow(file_path)

        await websocket.send_json({
            "type": "workflow_start",
            "workflow_name": workflow.name,
            "description": workflow.description,
            "nodes": [n.model_dump() for n in workflow.nodes],
            "task_prompt": task_prompt,
            "mock": mock_mode,
            "model": model,
        })

        llm = MockLLMClient() if mock_mode else GroqClient(default_model=model)
        executor = WorkflowExecutor(llm=llm)

        async def on_node_start(node_id: str):
            await websocket.send_json({
                "type": "node_start",
                "node_id": node_id,
            })

        async def on_tool_call(node_id: str, call: ToolCall):
            await websocket.send_json({
                "type": "tool_call",
                "node_id": node_id,
                "tool": {
                    "name": call.name,
                    "arguments": call.arguments,
                    "result": str(call.result) if call.result is not None else None,
                    "error": call.error,
                },
            })

        async def on_node_complete(node_id: str, result: NodeResult):
            await websocket.send_json({
                "type": "node_complete",
                "node_id": node_id,
                "result": {
                    "node_id": result.node_id,
                    "status": result.status.value,
                    "text": result.text,
                    "artifacts": result.artifacts,
                    "tool_calls": [
                        {
                            "name": tc.name,
                            "arguments": tc.arguments,
                            "result": str(tc.result) if tc.result is not None else None,
                            "error": tc.error,
                        }
                        for tc in result.tool_calls
                    ],
                    "error": result.error,
                },
            })

        run_result = await executor.run(
            workflow=workflow,
            task_prompt=task_prompt,
            on_node_start=on_node_start,
            on_tool_call=on_tool_call,
            on_node_complete=on_node_complete,
        )

        final_result = run_result.final_result
        final_text = final_result.text if final_result else ""
        final_node_id = run_result.final_node_id
        failed_nodes = [
            result.node_id
            for result in run_result.node_results.values()
            if result.status == NodeStatus.FAILED
        ]
        skipped_nodes = [
            result.node_id
            for result in run_result.node_results.values()
            if result.status == NodeStatus.SKIPPED
        ]
        workflow_status = "failed" if failed_nodes or (final_result and final_result.status != NodeStatus.SUCCESS) else "completed"

        await websocket.send_json({
            "type": "workflow_complete",
            "final_node_id": final_node_id,
            "final_text": final_text,
            "status": workflow_status,
            "failed_nodes": failed_nodes,
            "skipped_nodes": skipped_nodes,
        })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


if FRONTEND_DIST_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST_DIR), html=True), name="static")
