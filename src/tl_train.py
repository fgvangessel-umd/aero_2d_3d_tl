import torch
from torch.utils.data import DataLoader
from tl_model import AirfoilTransformerModel
from tl_data import create_dataloaders, AirfoilDataScaler
from experiment import ExperimentManager
from validation import ModelValidator, ValidationMetrics
from config import TrainingConfig
import logging
from pathlib import Path
from typing import Dict, Optional
import wandb
import argparse
from datetime import datetime
import logging
import sys


class ModelTrainer:
    def __init__(
        self,
        config: TrainingConfig,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        dataloaders: Dict[str, DataLoader],
        device: torch.device,
        scaler: Optional[AirfoilDataScaler] = None,
        experiment: Optional[ExperimentManager] = None,
        validator: Optional[ModelValidator] = None,
    ):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.dataloaders = dataloaders
        self.device = device
        self.scaler = scaler
        self.experiment = experiment
        self.validator = validator
        self.global_step = 0  # global steps for logging to wandb

        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.setup_logging()

    def setup_logging(self):
        """Configure logging"""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0

        for batch_idx, batch in enumerate(self.dataloaders["train"]):
            # Scale batch if scaler is provided
            if self.scaler:
                batch = self.scaler.transform(batch)

            # Move data to device
            airfoil_2d = batch["airfoil_2d"].to(self.device)
            geometry_3d = batch["geometry_3d"].to(self.device)
            pressure_3d = batch["pressure_3d"].to(self.device)
            mach = batch["mach"].to(self.device)
            reynolds = batch["reynolds"].to(self.device)
            z_coord = batch["z_coord"].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(airfoil_2d, geometry_3d, mach, reynolds, z_coord)

            # Compute loss and backward pass
            loss = self.criterion(predictions, pressure_3d)
            loss.backward()
            self.optimizer.step()

            # Update metrics
            total_loss += loss.item()

            ### DEBUG (REMOVE!!!!)
            sys.exit('DEBUG')

            # Log batch progress
            if batch_idx % 10 == 0:
                self.logger.info(
                    f"Epoch: {epoch} [{batch_idx}/{len(self.dataloaders['train'])}] "
                    f"Loss: {loss.item():.6f}"
                )

            # Log to W&B if experiment manager is available
            if self.experiment:
                self.experiment.log_batch_metrics(
                    {
                        "train/batch_loss": loss.item(),
                        "train/learning_rate": self.optimizer.param_groups[0]["lr"],
                    },
                    self.global_step,
                )
                self.global_step += 1

        return total_loss / len(self.dataloaders["train"])

    def run_validation(
        self, epoch: int, validation_type: str = "val"
    ) -> ValidationMetrics:
        """Run validation on the specified dataset"""
        if self.validator is None:
            self.logger.warning("No validator provided. Skipping validation.")
            return None

        self.logger.info(f"Running {validation_type} validation for epoch {epoch}")
        return self.validator.validate_dataset(
            self.dataloaders[validation_type], self.global_step
        )

    def train(self):
        """Main training loop"""
        best_val_loss = float("inf")

        """Define transfer learning epochs"""

        for epoch in range(self.config.num_epochs):
            self.logger.info(f"Starting epoch {epoch}")

            # Train for one epoch
            train_loss = self.train_epoch(epoch)

            # Periodic validation/Test
            metrics = {}
            if epoch % self.config.validation_freq == 0:
                # Validation set evaluation
                val_metrics = self.run_validation(epoch, "val")
                if val_metrics:
                    metrics.update(
                        {
                            "val/loss": val_metrics.mse,
                            "val/rmse": val_metrics.rmse,
                            "val/mae": val_metrics.mae,
                        }
                    )

                # Save best model
                if val_metrics.mse < best_val_loss:
                    best_val_loss = val_metrics.mse
                    if self.experiment:
                        self.experiment.save_checkpoint(
                            self.model, self.optimizer, epoch, metrics, self.scaler
                        )

                # Record training metrics
                train_metrics = self.run_validation(epoch, "train")
                if train_metrics:
                    metrics.update(
                        {
                            "train/loss": train_metrics.mse,
                            "train/rmse": train_metrics.rmse,
                            "train/mae": train_metrics.mae,
                        }
                    )

            # Test set evaluation if specified
            if hasattr(self.config, "test_freq") and epoch % self.config.test_freq == 0:
                test_metrics = self.run_validation(epoch, "test")
                if test_metrics:
                    metrics.update(
                        {
                            "test/loss": test_metrics.mse,
                            "test/rmse": test_metrics.rmse,
                            "test/mae": test_metrics.mae,
                        }
                    )

            # Log epoch metrics
            metrics.update({"train/loss": train_loss, "epoch": epoch})
            if self.experiment:
                self.experiment.log_epoch_metrics(metrics, self.global_step)

            # Regular model checkpoint
            if epoch % self.config.checkpoint_freq == 0 and self.experiment:
                self.experiment.save_checkpoint(
                    self.model, self.optimizer, epoch, metrics, self.scaler
                )

        self.logger.info("Training completed")
