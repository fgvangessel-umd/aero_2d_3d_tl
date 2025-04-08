import torch
from tl_train import ModelTrainer
from tl_model import AirfoilTransformerModel, AirfoilTransformerModel_TL_CrossAttention, AirfoilTransformerModel_Pretrain_Finetune
from tl_data import create_dataloaders, AirfoilDataScaler
from experiment import ExperimentManager
from validation import ModelValidator
from config import TrainingConfig
import logging
from datetime import datetime
import sys

def print_model_architecture(model):
    """
    Print the PyTorch model architecture including which layers have their weights frozen.
    
    Args:
        model: PyTorch model
    """
    print(f"Model Architecture: {model.__class__.__name__}")
    print("-" * 80)
    
    total_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        is_trainable = param.requires_grad
        
        print(f"Layer: {name}")
        print(f"  Shape: {tuple(param.shape)}")
        print(f"  Parameters: {param_count:,}")
        print(f"  Trainable: {'Yes' if is_trainable else 'No (Frozen)'}")
        print()
        
        total_params += param_count
        if is_trainable:
            trainable_params += param_count
    
    print("-" * 80)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,} ({trainable_params/total_params:.2%})")
    print(f"Frozen Parameters: {total_params - trainable_params:,} ({(total_params - trainable_params)/total_params:.2%})")


def train_model():
    """Main training function with improved configuration handling"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("main")

    # Load configuration from command line (with optional YAML base)
    config = TrainingConfig.from_args()

    # Set timestamp and initialize experiment directories and tracking
    experiment = ExperimentManager(config)
    experiment.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment.setup_directories()
    experiment.setup_wandb()

    # Set device based on configuration and availability
    if config.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(config.device)

    logging.info(f"Using device: {device}")

    # Create dataloaders
    dataloaders = create_dataloaders(
        config.data_path, batch_size=config.batch_size, num_workers=config.num_workers
    )

    # Initialize or load scaler
    scaler = AirfoilDataScaler()
    scaler.fit(dataloaders["train"])

    # Initialize model and training components
    if config.enable_transfer_learning:
        print("Transfer learning enabled")
        if hasattr(config, 'finetune') and config.finetune:
            print("Using pretrain-finetune transfer learning approach")
            model = AirfoilTransformerModel_Pretrain_Finetune(config).to(device)
        else:
            print("Using cross-attention transfer learning approach")
            model = AirfoilTransformerModel_TL_CrossAttention(config).to(device)
    else:
        print("Transfer learning disabled")
        model = AirfoilTransformerModel(config).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    criterion = torch.nn.MSELoss()

    # Initialize validator
    validator = ModelValidator(
        config=config,
        model=model,
        criterion=criterion,
        device=device,
        scaler=scaler,
        log_to_wandb=True,
    )

    # Initialize trainer
    trainer = ModelTrainer(
        config=config,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        dataloaders=dataloaders,
        device=device,
        scaler=scaler,
        experiment=experiment,
        validator=validator,
    )

    # Start training
    trainer.train()

    # Generate visualizations for last epoch (in the future exchange this for loading the best epoch)
    if experiment:
        for split in ["train", "val", "test"]:
            experiment.log_model_predictions(
                model,
                split,
                dataloaders[split],
                device,
                config.num_epochs,
                scaler,
                None,
            )

    # Cleanup
    if experiment:
        experiment.finish()


if __name__ == "__main__":
    # Train model
    train_model()