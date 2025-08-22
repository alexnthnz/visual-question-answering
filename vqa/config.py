"""Configuration management for VQA training and evaluation."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for VQA model architecture."""
    
    model_name: str = "openai/clip-vit-large-patch14"  # Upgraded to larger backbone for improved performance
    num_answers: int = 3000
    hidden_dim: int = 512
    dropout: float = 0.1
    unfreeze_clip: bool = False
    fusion_type: str = "concat"  # Options: 'concat', 'cross_attention'
    num_fusion_layers: int = 2
    num_attention_heads: int = 8
    use_lora: bool = False
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    
    epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    data_fraction: float = 0.1
    save_every_n_epochs: int = 2
    gradient_accumulation_steps: int = 1
    staged_unfreeze_epoch: int = 0  # Epoch to unfreeze CLIP (0 means never, unless unfreeze_clip=True from start)
    
    # Paths
    save_path: str = "models/vqa_model"
    vocab_path: str = "models/answer_vocab.json"
    
    
@dataclass
class EvaluationConfig:
    """Configuration for evaluation parameters."""
    
    batch_size: int = 32
    data_fraction: float = 0.1
    model_path: Optional[str] = None
    vocab_path: str = "models/answer_vocab.json"
    results_path: str = "evaluation_results.json"


@dataclass
class DataConfig:
    """Configuration for data handling."""
    
    data_dir: str = "data"
    dataset_name: str = "vqa"
    dataset_config: str = "vqa_v2"
    
    
def get_default_configs() -> tuple[ModelConfig, TrainingConfig, EvaluationConfig, DataConfig]:
    """Get default configurations for all components."""
    return (
        ModelConfig(),
        TrainingConfig(), 
        EvaluationConfig(),
        DataConfig()
    )


def ensure_directories() -> None:
    """Ensure all necessary directories exist."""
    directories = [
        Path("models"),
        Path("data"),
        Path("logs"),
        Path("results")
    ]
    
    for directory in directories:
        directory.mkdir(exist_ok=True)
