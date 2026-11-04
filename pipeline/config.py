"""Runner configuration. Secrets stay in this file or in the environment."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path("config/pipeline.json")


class ConfigError(RuntimeError):
    pass


@dataclass
class Limits:
    prepare_seconds: float = 120
    agent_seconds: float = 900
    tool_seconds: float = 60
    replay_seconds: float = 120


@dataclass
class DockerConfig:
    tools_image: str = "pipeline-tools:latest"
    task_cpus: float = 1.0
    task_memory: str = "256m"
    tools_memory: str = "512m"
    service_start_seconds: float = 30


@dataclass
class Config:
    provider: str
    model: str
    thinking: str = "off"
    api_key: str = ""
    api_key_env: str = ""
    auth_file: str = ""
    runs_dir: Path = Path("runs")
    tasks_dir: Path = Path("tasks")
    port: int = 8765
    tool_output_limit_bytes: int = 262144
    submission_limit_bytes: int = 10 * 1024 * 1024
    event_limit_bytes: int = 1024 * 1024
    limits: Limits = field(default_factory=Limits)
    docker: DockerConfig = field(default_factory=DockerConfig)

    def resolved_api_key(self) -> str:
        if self.api_key:
            return self.api_key
        if self.api_key_env:
            return os.environ.get(self.api_key_env, "")
        return ""

    def public(self) -> dict[str, Any]:
        """Configuration snapshot safe to store in the run package."""
        value = asdict(self)
        value.pop("api_key", None)
        value.pop("api_key_env", None)
        value["auth_file"] = "configured" if self.auth_file else ""
        value["runs_dir"] = str(self.runs_dir)
        value["tasks_dir"] = str(self.tasks_dir)
        return value


def load_config(path: Path | str | None = None) -> Config:
    raw_path = Path(path or os.environ.get("PIPELINE_CONFIG") or DEFAULT_CONFIG_PATH)
    if not raw_path.exists():
        raise ConfigError(
            f"config not found: {raw_path}; copy config/pipeline.example.json and set provider/model/api_key"
        )
    with raw_path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not data.get("provider") or not data.get("model"):
        raise ConfigError("config requires 'provider' and 'model'")
    limits = Limits(**data.get("limits", {}))
    docker = DockerConfig(**data.get("docker", {}))
    known = {
        "provider", "model", "thinking", "api_key", "api_key_env", "auth_file",
        "runs_dir", "tasks_dir", "port", "tool_output_limit_bytes",
        "submission_limit_bytes", "event_limit_bytes", "limits", "docker",
    }
    unknown = set(data) - known
    if unknown:
        raise ConfigError(f"unknown config keys: {sorted(unknown)}")
    cfg = Config(
        provider=data["provider"],
        model=data["model"],
        thinking=data.get("thinking", "off"),
        api_key=data.get("api_key", ""),
        api_key_env=data.get("api_key_env", ""),
        auth_file=data.get("auth_file", ""),
        runs_dir=Path(data.get("runs_dir", "runs")),
        tasks_dir=Path(data.get("tasks_dir", "tasks")),
        port=int(data.get("port", 8765)),
        tool_output_limit_bytes=int(data.get("tool_output_limit_bytes", 262144)),
        submission_limit_bytes=int(data.get("submission_limit_bytes", 10 * 1024 * 1024)),
        event_limit_bytes=int(data.get("event_limit_bytes", 1024 * 1024)),
        limits=limits,
        docker=docker,
    )
    if cfg.auth_file:
        cfg.auth_file = str(Path(cfg.auth_file).expanduser())
    return cfg
