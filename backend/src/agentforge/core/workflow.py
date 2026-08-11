"""Loads and validates a WorkflowConfig from a YAML file."""
from __future__ import annotations

import yaml

from agentforge.core.schema import WorkflowConfig


def load_workflow(path: str) -> WorkflowConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return WorkflowConfig.model_validate(raw)
