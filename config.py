import yaml
from dataclasses import dataclass, asdict
from dataclasses import dataclass, asdict
from typing import List, Optional
from datetime import datetime
import os

@dataclass
class TrainingConfig:
    # Model architecture
    d_model: int = 256
    n_head: int = 8
    n_layers: int = 6
    d_ff: int = 1024
    dropout: float = 0.1
    
    # Training parameters
    batch_size: int = 64
    learning_rate: float = 1e-4
    num_epochs: int = 100
    weight_decay: float = 0.0
    scheduler_type: str = 'cosine'
    warmup_steps: int = 1000
    checkpoint_freq: int = 10
    num_output_figs: int = 4
    
    # Data parameters
    data_path: str = 'data'
    num_workers: int = 4
    scaler_fname: str = 'airfoil_scaler.pt'
    
    # Experiment tracking
    project_name: str = 'airfoil-transfer-learning'
    experiment_name: str = 'baseline'
    tags: Optional[List[str]] = None

    # Wandb settings
    notes: str = "Baseline transformer model for transfer learning of 2D to 3D airfoil pressure prediction"
    
    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(asdict(self), f)
    
    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)