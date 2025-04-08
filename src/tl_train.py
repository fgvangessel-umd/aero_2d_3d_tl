import torch
from torch.utils.data import DataLoader
from tl_model import AirfoilTransformerModel, AirfoilTransformerModel_TL_CrossAttention, AirfoilTransformerModel_Pretrain_Finetune, PhaseType
from tl_data import create_dataloaders, AirfoilDataScaler, unpack_batch
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
            airfoil_2d, geometry_2d, pressure_2d, geometry_3d, pressure_3d, mach, reynolds, z_coord = \
                unpack_batch(batch, self.device)

            # Forward pass
            self.optimizer.zero_grad()
            
            # Handle different model types and phases
            if isinstance(self.model, AirfoilTransformerModel_Pretrain_Finetune):
                # For pretrain-finetune model, determine target based on phase
                if self.model.phase == PhaseType.PRETRAIN:
                    # In pretraining, use 2D airfoil pressure as target
                    # Note: we use only the pressure channel (last dimension) from airfoil_2d
                    pressure_target = pressure_2d.unsqueeze(-1)
                    predictions = self.model(geometry_2d, mach, reynolds, z_coord)
                else:  # FINETUNE phase
                    # In finetuning, use 3D pressure as target
                    pressure_target = pressure_3d
                    predictions = self.model(geometry_3d, mach, reynolds, z_coord)
            elif self.config.enable_transfer_learning:
                # Regular transfer learning model
                pressure_target = pressure_3d
                predictions = self.model(airfoil_2d, geometry_3d, mach, reynolds, z_coord)
            else:
                # Regular model without transfer learning
                pressure_target = pressure_3d
                predictions = self.model(geometry_3d, mach, reynolds, z_coord)

            # Compute loss and backward pass
            loss = self.criterion(predictions, pressure_target)
            loss.backward()
            self.optimizer.step()

            # Update metrics
            total_loss += loss.item()

            # Log batch progress
            if batch_idx % 10 == 0:
                phase = getattr(self.model, 'phase', None)
                phase_str = f"Phase: {phase.value} " if phase else ""
                self.logger.info(
                    f"Epoch: {epoch} [{batch_idx}/{len(self.dataloaders['train'])}] "
                    f"{phase_str}Loss: {loss.item():.6f}"
                )

            # Log to W&B if experiment manager is available
            if self.experiment:
                phase = getattr(self.model, 'phase', None)
                metrics = {
                    "train/batch_loss": loss.item(),
                    "train/learning_rate": self.optimizer.param_groups[0]["lr"],
                }
                if phase:
                    metrics["phase"] = phase.value
                    
                self.experiment.log_batch_metrics(metrics, self.global_step)
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
        """Main training loop with pretraining and finetuning phases"""
        
        # Check if we're using the pretrain-finetune model
        is_pretrain_finetune = isinstance(self.model, AirfoilTransformerModel_Pretrain_Finetune)
        
        if not self.config.enable_transfer_learning or not is_pretrain_finetune:
            # Regular training without the TL model
            self._train_standard(0, self.config.num_epochs)
            return
        
        # Get total number of epochs for each phase
        pretrain_epochs = self.config.pretrain_epochs
        finetune_epochs = self.config.finetune_epochs
        total_epochs = pretrain_epochs + finetune_epochs
        
        # Phase 1: Pretraining on 2D airfoil data
        self.logger.info("=== Starting pretraining phase ===")
        self.model.set_phase(PhaseType.PRETRAIN)
        self._train_standard(0, pretrain_epochs, phase="pretrain")
        
        # Save pretrained model checkpoint
        if self.experiment:
            self.experiment.save_checkpoint(
                self.model, self.optimizer, pretrain_epochs - 1, 
                {"phase": "pretrain_complete"}, self.scaler, 
                checkpoint_name="pretrained_model"
            )
        
        # Phase 2: Finetuning on 3D data with frozen layers
        self.logger.info("=== Starting finetuning phase ===")
        self.model.set_phase(PhaseType.FINETUNE)  # This will freeze layers
        self._train_standard(pretrain_epochs, total_epochs, phase="finetune")
        
        self.logger.info("Training completed")
    
    def _train_standard(self, start_epoch, end_epoch, phase="standard"):
        """Standard training procedure for either pretraining or finetuning"""
        best_val_loss = float("inf")
        
        for epoch in range(start_epoch, end_epoch):
            # Calculate relative epoch for logging
            rel_epoch = epoch - start_epoch
            self.logger.info(f"Starting {phase} epoch {rel_epoch + 1}/{end_epoch - start_epoch} (global: {epoch + 1})")

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
                            "phase": phase,
                        }
                    )

                # Save best model for this phase
                if val_metrics.mse < best_val_loss:
                    best_val_loss = val_metrics.mse
                    if self.experiment:
                        self.experiment.save_checkpoint(
                            self.model, self.optimizer, epoch, metrics, self.scaler,
                            checkpoint_name=f"best_{phase}_model"
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
            metrics.update({"train/loss": train_loss, "epoch": epoch, "phase": phase})
            if self.experiment:
                self.experiment.log_epoch_metrics(metrics, self.global_step)

            # Regular model checkpoint
            if epoch % self.config.checkpoint_freq == 0 and self.experiment:
                self.experiment.save_checkpoint(
                    self.model, self.optimizer, epoch, metrics, self.scaler,
                    checkpoint_name=f"{phase}_checkpoint_epoch_{rel_epoch}"
                )
