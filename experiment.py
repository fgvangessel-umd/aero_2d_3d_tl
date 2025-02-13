# experiment.py
import wandb
import torch
import os
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

class ExperimentManager:
    def __init__(self, config):
        self.config = config
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.setup_directories()
        self.setup_wandb()
        
    def setup_directories(self):
        """Create experiment directory structure"""
        self.exp_dir = Path(f"experiments/{self.config.experiment_name}_{self.timestamp}")
        self.model_dir = self.exp_dir / "models"
        self.viz_dir = self.exp_dir / "visualizations"
        self.log_dir = self.exp_dir / "logs"
        
        # Create directories
        for dir_path in [self.exp_dir, self.model_dir, self.viz_dir, self.log_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        # Save configuration
        self.config.save(self.exp_dir / "config.yaml")
        
    def setup_wandb(self):
        """Initialize W&B project"""
        self.run = wandb.init(
            project=self.config.project_name,
            name=f"{self.config.experiment_name}_{self.timestamp}",
            config=asdict(self.config),
            tags=self.config.tags
        )
        
    def log_batch_metrics(self, metrics, step):
        """Log training metrics for each batch"""
        wandb.log(metrics, step=step)
        
    def log_epoch_metrics(self, metrics, step):
        """Log metrics for each epoch"""
        metrics['step'] = step
        wandb.log(metrics)
        
        # Save metrics to local file
        metrics_file = self.log_dir / "metrics.jsonl"
        with open(metrics_file, "a") as f:
            f.write(json.dumps(metrics) + "\n")
            
    def log_model_predictions(self, model, dataloader, device, epoch, num_samples, scaler, global_step):
        """Generate and log model prediction visualizations"""
        model.eval()
        
        with torch.no_grad():

            for batch_idx, batch in enumerate(dataloader):

                # Get predictions (using your existing visualization code)
                fig = self.create_prediction_visualization(model, batch, device, num_samples, scaler)
                
                # Log to W&B
                wandb.log({
                    "predictions": wandb.Image(fig),
                    "epoch": epoch,
                    "batch": batch_idx,
                    "step": global_step
                })
            
                # Save locally
                fig.savefig(self.viz_dir / f"predictions_epoch_{epoch}_{batch_idx}.png")
                plt.close(fig)
            
    def create_prediction_visualization(self, model, batch, device, num_samples, scaler):
        """Create visualization of model predictions"""
        with torch.no_grad():

            # Scale batch
            batch = scaler.transform(batch)

            # Move data to device
            airfoil_2d = batch['airfoil_2d'].to(device)
            geometry_3d = batch['geometry_3d'].to(device)
            pressure_3d_true = batch['pressure_3d'].to(device)
            mach = batch['mach'].to(device)
            reynolds = batch['reynolds'].to(device)
            z_coord = batch['z_coord'].to(device)
            case_ids = batch['case_id']
            
            # Scale batch
            batch = scaler.transform(batch)
            
            # Get model predictions
            pressure_3d_pred = model(
                airfoil_2d,
                geometry_3d,
                mach,
                reynolds,
                z_coord
            )
            
            # Store ground-truth and prediction data
            batch_pred = {
                'pressure_3d': pressure_3d_pred,
                'airfoil_2d': airfoil_2d,
                'geometry_3d': geometry_3d,
                'mach': mach,
                'reynolds': reynolds
            }
            batch_true = {
                'pressure_3d': pressure_3d_true,
                'airfoil_2d': airfoil_2d,
                'geometry_3d': geometry_3d,
                'mach': mach,
                'reynolds': reynolds
            }
            
            # Visualize random samples
            for i in range(min(num_samples, len(case_ids))):
                fig = plt.figure(figsize=(20, 15))
                gs = GridSpec(2, 2, figure=fig)
                
                # 1. Geometric Configuration (2D & 3D)
                ax_geo = fig.add_subplot(gs[0, 0])
                x_2d = airfoil_2d[i,:,1].cpu().numpy()
                y_2d = airfoil_2d[i,:,2].cpu().numpy()
                x_3d = geometry_3d[i,:,1].cpu().numpy()
                y_3d = geometry_3d[i,:,2].cpu().numpy()
                
                ax_geo.scatter(x_2d, y_2d, c='blue', label='2D Profile', alpha=0.6)
                ax_geo.scatter(x_3d, y_3d, c='red', label='3D Section', alpha=0.6)
                ax_geo.set_title(f'Geometric Configuration\nz = {z_coord[i].item():.3f}', fontsize=20)
                ax_geo.set_aspect('equal')
                ax_geo.legend(fontsize=18)
                ax_geo.grid(True)
                
                # 2. Pressure Distribution
                ax_press = fig.add_subplot(gs[0, 1])
                p_2d = batch_true['airfoil_2d'][i,:,3].cpu().numpy()
                p_3d_true = batch_true['pressure_3d'][i,:,0].cpu().numpy()
                p_3d_pred = batch_pred['pressure_3d'][i,:,0].cpu().numpy()
                
                ax_press.plot(x_2d, p_2d, c='k', linestyle='-.', label='2D Pressure', alpha=0.6)
                ax_press.scatter(x_3d, p_3d_true, c='black', label='3D True', alpha=0.6)
                ax_press.scatter(x_3d, p_3d_pred, c='red', label='3D Predicted', alpha=0.6)
                ax_press.set_title('Pressure Distribution', fontsize=20)
                ax_press.legend(fontsize=18)
                ax_press.grid(True)
                
                # 3. Pressure Difference (True vs Predicted)
                ax_diff = fig.add_subplot(gs[1, 0])
                pressure_diff = p_3d_pred - p_3d_true
                ax_diff.scatter(x_3d, pressure_diff, c='purple', alpha=0.6)
                ax_diff.set_title('Prediction Error', fontsize=20)
                ax_diff.axhline(y=0, color='k', linestyle='--')
                ax_diff.grid(True)
                
                # 4. Error Distribution
                ax_hist = fig.add_subplot(gs[1, 1])
                ax_hist.hist(pressure_diff, bins=30, alpha=0.6, color='purple')
                ax_hist.set_title('Error Distribution', fontsize=20)
                ax_hist.axvline(x=0, color='k', linestyle='--')
                ax_hist.grid(True)
                
                # Add case information
                fig.suptitle(f'Case ID: {case_ids[i].item()}\n' + 
                            f'Mach: {mach[i].item():.3f}, ' +
                            f'Reynolds: {batch_true["reynolds"][i].item():.2e}',
                            fontsize=24)

        return fig
    
    def save_checkpoint(self, model, optimizer, epoch, metrics):
        """Save model checkpoint with metadata"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'config': asdict(self.config)
        }
        
        # Save locally
        checkpoint_path = self.model_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Log to W&B
        artifact = wandb.Artifact(
            name=f'model-checkpoint-{epoch}',
            type='model',
            metadata=metrics
        )
        artifact.add_file(str(checkpoint_path))
        self.run.log_artifact(artifact)
        
    def finish(self):
        """Cleanup and finish experiment tracking"""
        wandb.finish()