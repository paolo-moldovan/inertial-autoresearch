"""Typed configuration dataclasses for all experiment settings."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DeviceConfig:
    """Hardware and compute settings."""

    preferred: str = "auto"  # "auto" | "cuda" | "mps" | "cpu"
    dtype: str = "float32"  # "float32" | "float16" | "bfloat16"
    compile: bool = False  # torch.compile (not supported on MPS)


@dataclass(frozen=True)
class DataSubsetConfig:
    """Reproducible split-local data subsetting for quick experiments."""

    enabled: bool = False
    seed: int = 42
    train_sequence_fraction: float = 1.0
    val_sequence_fraction: float = 1.0
    test_sequence_fraction: float = 1.0
    train_max_sequences: int | None = None
    val_max_sequences: int | None = None
    test_max_sequences: int | None = None
    train_window_fraction: float = 1.0
    val_window_fraction: float = 1.0
    test_window_fraction: float = 1.0
    train_max_windows: int | None = None
    val_max_windows: int | None = None
    test_max_windows: int | None = None


@dataclass(frozen=True)
class DataConfig:
    """Dataset and preprocessing settings."""

    dataset: str = "euroc"  # "euroc" | "blackbird" | "synthetic"
    sequences: list[str] = field(default_factory=list)  # empty = all available
    window_size: int = 200  # samples per window (1s at 200Hz for EuRoC)
    stride: int = 100  # 50% overlap by default
    normalize: bool = True
    augment: bool = True
    dataset_kwargs: dict[str, object] = field(default_factory=dict)
    train_sequences: list[str] = field(default_factory=list)
    val_sequences: list[str] = field(default_factory=list)
    test_sequences: list[str] = field(default_factory=list)
    subset: DataSubsetConfig = field(default_factory=DataSubsetConfig)
    data_dir: str = "data"


@dataclass(frozen=True)
class ModelConfig:
    """Model architecture settings."""

    name: str = "lstm"  # registered model name
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    bidirectional: bool = True
    # Transformer-specific
    num_heads: int = 4
    # Conv1D-specific
    kernel_size: int = 7
    dilation_base: int = 2
    # Extra kwargs passed to model constructor
    extra: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingConfig:
    """Training loop settings."""

    epochs: int = 100
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    scheduler: str = "cosine"  # "cosine" | "step" | "plateau" | "none"
    scheduler_kwargs: dict[str, object] = field(default_factory=dict)
    optimizer: str = "adamw"  # "adam" | "adamw" | "sgd"
    loss: str = "mse"  # "mse" | "huber" | "spectral"
    gradient_clip: float = 1.0
    early_stop_patience: int = 10
    seed: int = 42
    num_workers: int = 4
    time_budget_sec: int = 0  # 0 = no budget


@dataclass(frozen=True)
class HermesConfig:
    """Hermes-backed orchestration settings for local LLM-driven search."""

    python_bin: str = "vendor/hermes-agent/.venv/bin/python"
    cli_path: str = "vendor/hermes-agent/cli.py"
    home_dir: str = ".hermes"
    provider: str = "custom"
    base_url: str = "http://127.0.0.1:11434/v1"
    model: str = "qwen3.5:latest"
    api_key: str | None = None
    toolsets: list[str] = field(default_factory=lambda: ["file"])
    max_turns: int = 6
    timeout_sec: int = 90
    healthcheck_timeout_sec: float = 2.0


@dataclass(frozen=True)
class AutoResearchConfig:
    """Auto-research loop settings."""

    max_iterations: int = 50
    time_budget_sec: int = 600
    metric_key: str = "val_rmse"
    metric_direction: str = "minimize"  # "minimize" | "maximize"
    results_file: str = "artifacts/autoresearch/results.tsv"
    orchestrator: str = "none"  # "none" | "hermes" | "researchclaw"
    hermes: HermesConfig = field(default_factory=HermesConfig)


@dataclass(frozen=True)
class ObservabilityConfig:
    """Mission-control observability settings."""

    enabled: bool = True
    db_path: str = "artifacts/observability/mission_control.db"
    blob_dir: str = "artifacts/observability/blobs"
    capture_raw_llm: bool = True
    import_hermes_state: bool = True
    redact_secrets: bool = True
    tui_refresh_hz: int = 2
    streamlit_port: int = 8501
    mlflow_enabled: bool = False
    mlflow_tracking_uri: str = "file:./artifacts/mlruns"
    mlflow_experiment_name: str = "imu-mission-control"
    mlflow_log_artifacts: bool = True
    phoenix_enabled: bool = False
    phoenix_project_name: str = "imu-mission-control"
    phoenix_endpoint: str = "http://localhost:6006/v1/traces"
    phoenix_protocol: str = "http/protobuf"
    phoenix_batch: bool = False


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level experiment configuration combining all sub-configs."""

    name: str = "default"
    device: DeviceConfig = field(default_factory=DeviceConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    autoresearch: AutoResearchConfig = field(default_factory=AutoResearchConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    output_dir: str = "artifacts"
    log_dir: str = "artifacts/logs"

    @property
    def runs_dir(self) -> Path:
        return Path(self.output_dir) / "runs"

    @property
    def checkpoint_dir(self) -> Path:
        return Path(self.output_dir) / "checkpoints" / self.name

    @property
    def figures_dir(self) -> Path:
        return Path(self.output_dir) / "figures" / self.name
