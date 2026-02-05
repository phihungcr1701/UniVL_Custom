"""Configuration management using YAML files"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import yaml


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    # Encoder layers
    text_num_hidden_layers: int = 12
    visual_num_hidden_layers: int = 6
    cross_num_hidden_layers: int = 2
    decoder_num_hidden_layers: int = 3
    
    # Hidden sizes
    hidden_size: int = 768
    video_dim: int = 1024
    vocab_size: int = 30522
    
    # Sequence lengths
    max_words: int = 48
    max_frames: int = 48
    max_position_embeddings: int = 512
    
    # Dropout
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    
    # Pretrain mode
    pretrain_mode: bool = False


@dataclass
class DataConfig:
    """Data loading configuration"""
    # Paths
    data_dir: str = "data/msrvtt"
    train_csv: str = "MSRVTT_train.9k.csv"
    val_csv: str = "MSRVTT_JSFUSION_test.csv"
    data_json: str = "MSRVTT_data.json"
    features_path: str = "msrvtt_videos_features.pickle"
    
    # Data processing
    max_words: int = 48
    max_frames: int = 48
    num_workers: int = 4
    prefetch_factor: int = 2
    
    # MLM/MFM for pretrain
    mlm_probability: float = 0.15
    mfm_probability: float = 0.15


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Optimization
    learning_rate: float = 3e-5
    coef_lr: float = 0.1  # LR coefficient for BERT vs other modules
    weight_decay: float = 0.01
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    # Schedule
    epochs: int = 5
    warmup_ratio: float = 0.1
    
    # Batch sizes
    batch_size: int = 128
    batch_size_val: int = 32
    gradient_accumulation_steps: int = 1
    
    # Mixed precision
    use_amp: bool = True
    
    # Distributed training
    local_rank: int = -1
    n_gpu: int = 1
    
    # Checkpointing
    save_steps: int = 1000
    eval_steps: int = 1000
    logging_steps: int = 100
    
    # Output
    output_dir: str = "checkpoints"
    
    # Pretrain specific
    do_pretrain: bool = False


@dataclass
class EvaluationConfig:
    """Evaluation configuration"""
    # Generation
    beam_size: int = 5
    max_gen_length: int = 20
    length_penalty: float = 1.0
    
    # Metrics
    metrics: list = field(default_factory=lambda: ["bleu", "meteor", "rouge", "cider"])


@dataclass
class LoggingConfig:
    """Logging configuration"""
    # Logging backend
    use_wandb: bool = True
    use_tensorboard: bool = True
    
    # W&B settings
    wandb_project: str = "univl-msrvtt"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    
    # TensorBoard
    tensorboard_dir: str = "runs"
    
    # Logging level
    log_level: str = "INFO"


@dataclass
class UniVLConfig:
    """Complete UniVL configuration"""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Task type
    task_type: str = "caption"  # or "pretrain"
    stage_two: bool = True  # Always True for caption task
    
    # Random seed
    seed: int = 42
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "UniVLConfig":
        """Load configuration from YAML file"""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Create config objects
        model_config = ModelConfig(**config_dict.get('model', {}))
        data_config = DataConfig(**config_dict.get('data', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        eval_config = EvaluationConfig(**config_dict.get('evaluation', {}))
        logging_config = LoggingConfig(**config_dict.get('logging', {}))
        
        # Create main config
        config = cls(
            model=model_config,
            data=data_config,
            training=training_config,
            evaluation=eval_config,
            logging=logging_config,
            task_type=config_dict.get('task_type', 'caption'),
            stage_two=config_dict.get('stage_two', True),
            seed=config_dict.get('seed', 42),
        )
        
        # Sync max_words and max_frames between model and data
        config.model.max_words = config.data.max_words
        config.model.max_frames = config.data.max_frames
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'model': self.model.__dict__,
            'data': self.data.__dict__,
            'training': self.training.__dict__,
            'evaluation': self.evaluation.__dict__,
            'logging': self.logging.__dict__,
            'task_type': self.task_type,
            'stage_two': self.stage_two,
            'seed': self.seed,
        }
    
    def save_yaml(self, yaml_path: str):
        """Save configuration to YAML file"""
        os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


def load_config(config_path: str, overrides: Optional[Dict[str, Any]] = None) -> UniVLConfig:
    """
    Load configuration from YAML file with optional overrides
    
    Args:
        config_path: Path to YAML config file
        overrides: Dictionary of values to override (e.g., {'training.learning_rate': 1e-4})
    
    Returns:
        UniVLConfig object
    """
    config = UniVLConfig.from_yaml(config_path)
    
    if overrides:
        for key, value in overrides.items():
            # Support nested keys like 'training.learning_rate'
            parts = key.split('.')
            obj = config
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], value)
    
    return config
