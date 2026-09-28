"""Training configuration for Graph Autoencoder."""
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class DataConfig:
    """Data-related configuration."""
    dataset_name: str = "PBMC-Multiome"
    base_data_dir: str = "data/processed"
    modalities: tuple = ("Peaks", "RNA")
    batch_size: int = 256


@dataclass(frozen=True)
class ModelConfig:
    """Model architecture configuration."""
    hidden_channels: int = 512
    latent_channels: int = 512
    num_layers: int = 2
    num_clusters: int = 20
    clustering_weight: float = 0.01


@dataclass(frozen=True)
class TrainingConfig:
    """Training hyperparameters configuration."""
    learning_rate: float = 1e-3
    n_epochs: int = 500  # Change to 500 for full training
    warmup_epochs: int = 3
    seed: int = 42


@dataclass(frozen=True)
class OutputConfig:
    """Output directory configuration."""
    checkpoint_dir: str = "checkpoints"
    embedding_dir: str = "results/embedding"


@dataclass(frozen=True)
class Config:
    """Complete training configuration."""
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    output: OutputConfig = OutputConfig()


def get_default_config() -> Config:
    """Get default configuration instance.

    Returns:
        Default Config instance.
    """
    return Config()
