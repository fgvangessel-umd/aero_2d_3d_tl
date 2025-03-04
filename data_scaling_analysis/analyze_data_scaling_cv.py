import os
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import sys
from collections import defaultdict
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_absolute_error
from scipy import stats

def extract_experiment_info(exp_dir):
    """Extract fold number and training percentage from experiment directory name"""
    match = re.match(r'data_split_(\d+)_(\d+)_\d+', exp_dir)
    if match:
        fold = int(match.group(1))
        train_percent = int(match.group(2))
        return fold, train_percent
    return None, None

def load_metrics(exp_path):
    """Load metrics from the jsonl file"""
    metrics_path = os.path.join(exp_path, "logs", "metrics.jsonl")
    if not os.path.exists(metrics_path):
        print(f"Metrics file not found: {metrics_path}")
        return []
    
    metrics = []
    with open(metrics_path, 'r') as f:
        for line in f:
            try:
                metrics.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                print(f"Error parsing line in {metrics_path}")
    
    return metrics

def find_best_model(metrics):
    """Find the epoch with the best validation loss"""
    if not metrics:
        return None
    
    best_idx = -1
    best_val_loss = float('inf')
    
    for i, metric in enumerate(metrics):
        if 'val/loss' in metric and metric['val/loss'] < best_val_loss:
            best_val_loss = metric['val/loss']
            best_idx = i
    
    return metrics[best_idx] if best_idx >= 0 else None

def load_best_checkpoint_and_evaluate(experiment_dir, fold, train_percent, best_epoch):
    """
    Load the best checkpoint model and evaluate lift and drag predictions
    
    Args:
        experiment_dir: Path to the experiment directory
        fold: The fold number
        train_percent: The training data percentage
        best_epoch: The best epoch for this experiment
        
    Returns:
        Dict with lift and drag evaluation metrics
    """
    # Import required modules from your codebase
    # Add the current directory to sys.path to import modules
    current_dir = os.path.abspath('.')
    if current_dir not in sys.path:
        sys.path.append(current_dir)
        
    # Import required modules from your codebase
    from tl_model import AirfoilTransformerModel
    from tl_data import create_dataloaders, AirfoilDataScaler
    from config import TrainingConfig
    from utils import load_checkpoint, calculate_airfoil_forces, select_case
    from validation import ModelValidator
    
    # Load configuration
    config_path = os.path.join(experiment_dir, "config.yaml")
    config = TrainingConfig.load(config_path)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Construct data path based on fold and train percentage
    data_path = f"data_cv_splits/data_split_{fold}_{train_percent}"
    if not os.path.exists(data_path):
        print(f"Data path not found: {data_path}")
        return None
    
    # Create dataloaders for this specific data split
    dataloaders = create_dataloaders(
        data_path,
        batch_size=1024,
        num_workers=1  # Reduced for stability
    )
    
    # Initialize model and optimizer
    model = AirfoilTransformerModel(config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    criterion = torch.nn.MSELoss()
    
    # Initialize scaler
    scaler = AirfoilDataScaler()
    
    # Construct checkpoint path
    checkpoint_path = os.path.join(experiment_dir, "models", f"checkpoint_epoch_{best_epoch}.pt")
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint file not found: {checkpoint_path}")
        return None
    
    # Load checkpoint
    model, optimizer, scaler, epoch, _ = load_checkpoint(checkpoint_path, model, optimizer, scaler)
    
    print(f"Successfully loaded checkpoint from {checkpoint_path}")
    
    # Initialize validator
    validator = ModelValidator(
        model=model,
        criterion=criterion,
        device=device,
        scaler=scaler,
        log_to_wandb=False
    )
    
    model.eval()

    with torch.no_grad():

        # Get test cases
        for batch_idx, batch in enumerate(dataloaders['test']):
            # Move data to device
            test_cases = batch['case_id'].numpy().tolist()
        test_cases = list(set(test_cases))

        case_ids = {'test': test_cases}
        true_lift = []
        pred_lift = []
        true_drag = []
        pred_drag = []
    
        # Evaluate on  test datasets
        results = {}
    
        true_lift = []
        pred_lift = []
        true_drag = []
        pred_drag = []
        
        # Load data
        for split in ['test']:
            for case_id in case_ids[split]:

                # Process each batch in the dataloader
                for batch_idx, batch in enumerate(dataloaders[split]):
                    with torch.no_grad():

                        true_reynolds = batch['reynolds'].to(device)
                        batch = scaler.transform(batch)
                        
                        airfoil_2d, geometry_3d, pressure_3d, mach, reynolds, true_reynolds, z_coord = \
                        select_case(batch, true_reynolds, case_id, device)
                    
                        # Make model predictions
                        predictions = model(
                            airfoil_2d,
                            geometry_3d,
                            mach,
                            reynolds,
                            z_coord
                        )

                        # Convert data formats
                        xy_2d = airfoil_2d[:,:,1:3].cpu().numpy()
                        xy_3d = geometry_3d[:,:,1:3].cpu().numpy()
                        z_coord = z_coord.cpu().numpy()
                        p_2d = airfoil_2d[:,:,3].cpu().numpy()
                        p_3d_true = pressure_3d.cpu().numpy()
                        p_3d_pred = predictions.cpu().numpy()
                        
                        force_metrics = validator.compute_force_metrics(
                                predictions, 
                                pressure_3d, 
                                geometry_3d[:,:,1:3], 
                                alpha=2.5
                                )
                    
                        true_lift.append(force_metrics['true_wing_lift'])
                        pred_lift.append(force_metrics['pred_wing_lift'])
                        true_drag.append(force_metrics['true_wing_drag'])
                        pred_drag.append(force_metrics['pred_wing_drag'])
                
            # Calculate metrics
            corr_lift, _ = stats.pearsonr(true_lift, pred_lift)
            corr_drag, _ = stats.pearsonr(true_drag, pred_drag)
            
            r2_lift = r2_score(true_lift, pred_lift)
            r2_drag = r2_score(true_drag, pred_drag)
            
            mae_lift = mean_absolute_error(true_lift, pred_lift)
            mae_drag = mean_absolute_error(true_drag, pred_drag)
            
            mape_lift = mean_absolute_percentage_error(true_lift, pred_lift)
            mape_drag = mean_absolute_percentage_error(true_drag, pred_drag)
            
            # Store results
            results[split] = {
                'corr_lift': corr_lift,
                'corr_drag': corr_drag,
                'r2_lift': r2_lift,
                'r2_drag': r2_drag,
                'mae_lift': mae_lift,
                'mae_drag': mae_drag,
                'mape_lift': mape_lift,
                'mape_drag': mape_drag,
                'true_lift': true_lift,
                'pred_lift': pred_lift,
                'true_drag': true_drag,
                'pred_drag': pred_drag
            }
            
            print(f"  {split.upper()} results for {experiment_dir}:")
            print(f"    Correlation Lift: {corr_lift:.4f}")
            print(f"    Correlation Drag: {corr_drag:.4f}")
            print(f"    R2 Lift: {r2_lift:.4f}")
            print(f"    R2 Drag: {r2_drag:.4f}")
            print(f"    MAE Lift: {mae_lift:.6f}")
            print(f"    MAE Drag: {mae_drag:.6f}")
            print(f"    MAPE Lift: {mape_lift:.4f}")
            print(f"    MAPE Drag: {mape_drag:.4f}")
        
    return results


def main():
    # Define the base experiments directory
    experiments_dir = "experiments"
    
    # Dictionary to store results by training percentage
    results_by_percent = defaultdict(list)
    
    # Dictionary to store detailed results for each experiment
    detailed_results = []
    
    # Store lift/drag evaluation results
    force_evaluation_results = []
    
    # Track missing checkpoints
    missing_checkpoints = []
    
    # Function to check if checkpoint exists
    def check_checkpoint_exists(exp_path, epoch):
        """Check if a checkpoint file exists for the given epoch"""
        checkpoint_path = os.path.join(exp_path, "models", f"checkpoint_epoch_{epoch}.pt")
        exists = os.path.isfile(checkpoint_path)
        return exists, checkpoint_path
    
    # Process each experiment directory
    for exp_dir in os.listdir(experiments_dir):
        exp_path = os.path.join(experiments_dir, exp_dir)
        if not os.path.isdir(exp_path):
            continue
        
        fold, train_percent = extract_experiment_info(exp_dir)
        if fold is None or train_percent is None:
            print(f"Skipping directory with unexpected format: {exp_dir}")
            continue
        
        print(f"Processing experiment: Fold {fold}, Training {train_percent}%")
        
        # Load metrics from jsonl file
        metrics = load_metrics(exp_path)
        if not metrics:
            print(f"No metrics found for {exp_dir}")
            continue
        
        # Find the best model based on validation loss
        best_model = find_best_model(metrics)
        if best_model is None:
            print(f"No valid metrics found for {exp_dir}")
            continue
        
        # Check if checkpoint exists for best epoch
        best_epoch = best_model.get('epoch', -1)
        checkpoint_exists, checkpoint_path = check_checkpoint_exists(exp_path, best_epoch)
        
        if not checkpoint_exists:
            print(f"WARNING: Missing checkpoint for {exp_dir} at epoch {best_epoch}")
            missing_checkpoints.append({
                'experiment': exp_dir,
                'fold': fold,
                'train_percent': train_percent,
                'best_epoch': best_epoch,
                'expected_path': checkpoint_path
            })
        else:
            print(f"Found checkpoint for {exp_dir} at epoch {best_epoch}")
            
            # Evaluate model on lift and drag predictions
            print(f"Evaluating model for {exp_dir} (fold {fold}, train {train_percent}%, epoch {best_epoch})")
            force_results = load_best_checkpoint_and_evaluate(
                exp_path, 
                fold, 
                train_percent, 
                best_epoch
            )
            
            if force_results:
                # Store the aerodynamic force evaluation results
                force_result_entry = {
                    'experiment': exp_dir,
                    'fold': fold,
                    'train_percent': train_percent,
                    'epoch': best_epoch,
                    'val_loss': best_model.get('val/loss', float('inf')),
                    'test_loss': best_model.get('test/loss', float('inf')),
                    'test_rmse': best_model.get('test/rmse', float('inf')),
                    'test_mae': best_model.get('test/mae', float('inf')),
                    #'val_corr_lift': force_results['val']['corr_lift'],
                    #'val_corr_drag': force_results['val']['corr_drag'],
                    #'val_r2_lift': force_results['val']['r2_lift'],
                    #'val_r2_drag': force_results['val']['r2_drag'],
                    #'val_mae_lift': force_results['val']['mae_lift'],
                    #'val_mae_drag': force_results['val']['mae_drag'],
                    #'val_mape_lift': force_results['val']['mape_lift'],
                    #'val_mape_drag': force_results['val']['mape_drag'],
                    'test_corr_lift': force_results['test']['corr_lift'],
                    'test_corr_drag': force_results['test']['corr_drag'],
                    'test_r2_lift': force_results['test']['r2_lift'],
                    'test_r2_drag': force_results['test']['r2_drag'],
                    'test_mae_lift': force_results['test']['mae_lift'],
                    'test_mae_drag': force_results['test']['mae_drag'],
                    'test_mape_lift': force_results['test']['mape_lift'],
                    'test_mape_drag': force_results['test']['mape_drag'],
                }
                force_evaluation_results.append(force_result_entry)
        
        # Create detailed result entry
        result_entry = {
            'experiment': exp_dir,
            'fold': fold,
            'train_percent': train_percent,
            'epoch': best_model.get('epoch', -1),
            'val_loss': best_model.get('val/loss', float('inf')),
            'test_loss': best_model.get('test/loss', float('inf')),
            'test_rmse': best_model.get('test/rmse', float('inf')),
            'test_mae': best_model.get('test/mae', float('inf')),
            'checkpoint_exists': checkpoint_exists,
            'checkpoint_path': checkpoint_path if checkpoint_exists else None
        }
        
        # Store the results
        results_by_percent[train_percent].append(result_entry)
        detailed_results.append(result_entry)
    
    # Calculate aggregate statistics for each training percentage
    summary = []
    for percent, results in sorted(results_by_percent.items()):
        if not results:
            continue
        
        # Extract test metrics and best epoch information
        test_losses = [r['test_loss'] for r in results if 'test_loss' in r]
        test_rmses = [r['test_rmse'] for r in results if 'test_rmse' in r]
        test_maes = [r['test_mae'] for r in results if 'test_mae' in r]
        best_epochs = [r['epoch'] for r in results if 'epoch' in r]
        
        # Calculate statistics
        summary.append({
            'train_percent': percent,
            'num_folds': len(results),
            'best_epoch_mean': np.mean(best_epochs) if best_epochs else float('nan'),
            'best_epoch_min': np.min(best_epochs) if best_epochs else float('nan'),
            'best_epoch_max': np.max(best_epochs) if best_epochs else float('nan'),
            'best_epoch_std': np.std(best_epochs) if best_epochs else float('nan'),
            'test_loss_mean': np.mean(test_losses) if test_losses else float('nan'),
            'test_loss_std': np.std(test_losses) if test_losses else float('nan'),
            'test_rmse_mean': np.mean(test_rmses) if test_rmses else float('nan'),
            'test_rmse_std': np.std(test_rmses) if test_rmses else float('nan'),
            'test_mae_mean': np.mean(test_maes) if test_maes else float('nan'),
            'test_mae_std': np.std(test_maes) if test_maes else float('nan'),
        })
    
    # Create a DataFrame for better display
    summary_df = pd.DataFrame(summary)
    print("\nSummary of Test Performance by Training Data Percentage:")
    print(summary_df)
    
    # Save force evaluation results
    if force_evaluation_results:
        force_df = pd.DataFrame(force_evaluation_results)
        force_df.to_csv('force_evaluation_results.csv', index=False)
        print("Force evaluation results saved to force_evaluation_results.csv")
        
        # Calculate aggregated force metrics by training percentage
        force_summary = []
        for percent, group in force_df.groupby('train_percent'):
            force_summary.append({
                'train_percent': percent,
                'num_models': len(group),
                'test_corr_lift_mean': group['test_corr_lift'].mean(),
                'test_corr_lift_std': group['test_corr_lift'].std(),
                'test_corr_drag_mean': group['test_corr_drag'].mean(),
                'test_corr_drag_std': group['test_corr_drag'].std(),
                'test_r2_lift_mean': group['test_r2_lift'].mean(),
                'test_r2_lift_std': group['test_r2_lift'].std(),
                'test_r2_drag_mean': group['test_r2_drag'].mean(),
                'test_r2_drag_std': group['test_r2_drag'].std(),
                'test_mae_lift_mean': group['test_mae_lift'].mean(),
                'test_mae_lift_std': group['test_mae_lift'].std(),
                'test_mae_drag_mean': group['test_mae_drag'].mean(),
                'test_mae_drag_std': group['test_mae_drag'].std(),
                'test_mape_lift_mean': group['test_mape_lift'].mean(),
                'test_mape_lift_std': group['test_mape_lift'].std(),
                'test_mape_drag_mean': group['test_mape_drag'].mean(),
                'test_mape_drag_std': group['test_mape_drag'].std(),
            })
        
        force_summary_df = pd.DataFrame(force_summary)
        force_summary_df = force_summary_df.sort_values('train_percent')
        force_summary_df.to_csv('force_metrics_summary.csv', index=False)
        print("Force metrics summary saved to force_metrics_summary.csv")
        
        # Create visualization of force prediction metrics
        plt.figure(figsize=(15, 12))
        
        # Plot correlation coefficient for lift
        plt.subplot(2, 2, 1)
        plt.errorbar(
            force_summary_df['train_percent'],
            force_summary_df['test_corr_lift_mean'],
            yerr=force_summary_df['test_corr_lift_std'],
            fmt='o-',
            capsize=5,
            color='blue'
        )
        plt.xlabel('Training Data Percentage')
        plt.ylabel('Correlation Coefficient')
        plt.title('Lift Prediction Correlation')
        plt.grid(True)
        
        # Plot correlation coefficient for drag
        plt.subplot(2, 2, 2)
        plt.errorbar(
            force_summary_df['train_percent'],
            force_summary_df['test_corr_drag_mean'],
            yerr=force_summary_df['test_corr_drag_std'],
            fmt='o-',
            capsize=5,
            color='red'
        )
        plt.xlabel('Training Data Percentage')
        plt.ylabel('Correlation Coefficient')
        plt.title('Drag Prediction Correlation')
        plt.grid(True)
        
        # Plot R² for lift
        plt.subplot(2, 2, 3)
        plt.errorbar(
            force_summary_df['train_percent'],
            force_summary_df['test_r2_lift_mean'],
            yerr=force_summary_df['test_r2_lift_std'],
            fmt='o-',
            capsize=5,
            color='blue'
        )
        plt.xlabel('Training Data Percentage')
        plt.ylabel('R² Score')
        plt.title('Lift Prediction R²')
        plt.grid(True)
        
        # Plot R² for drag
        plt.subplot(2, 2, 4)
        plt.errorbar(
            force_summary_df['train_percent'],
            force_summary_df['test_r2_drag_mean'],
            yerr=force_summary_df['test_r2_drag_std'],
            fmt='o-',
            capsize=5,
            color='red'
        )
        plt.xlabel('Training Data Percentage')
        plt.ylabel('R² Score')
        plt.title('Drag Prediction R²')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('force_prediction_performance.png')
        print("Force prediction performance visualization saved to force_prediction_performance.png")
        
        # Create a second visualization for MAE metrics
        plt.figure(figsize=(15, 8))
        
        # Plot MAE for lift
        plt.subplot(1, 2, 1)
        plt.errorbar(
            force_summary_df['train_percent'],
            force_summary_df['test_mae_lift_mean'],
            yerr=force_summary_df['test_mae_lift_std'],
            fmt='o-',
            capsize=5,
            color='blue'
        )
        plt.xlabel('Training Data Percentage')
        plt.ylabel('Mean Absolute Error')
        plt.title('Lift Prediction MAE')
        plt.grid(True)
        
        # Plot MAE for drag
        plt.subplot(1, 2, 2)
        plt.errorbar(
            force_summary_df['train_percent'],
            force_summary_df['test_mae_drag_mean'],
            yerr=force_summary_df['test_mae_drag_std'],
            fmt='o-',
            capsize=5,
            color='red'
        )
        plt.xlabel('Training Data Percentage')
        plt.ylabel('Mean Absolute Error')
        plt.title('Drag Prediction MAE')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('force_prediction_mae.png')
        #print("Force prediction MAE visualization saved to force_prediction_mae.png",
        #        'test_mape_lift_mean': group['test_mape_lift'].mean(),
        #        'test_mape_lift_std': group['test_mape_lift'].std(),
        #        'test_mape_drag_mean': group['test_mape_drag'].mean(),
        #        'test_mape_drag_std': group['test_mape_drag'].std(),
        #    )
        
        force_summary_df = pd.DataFrame(force_summary)
        force_summary_df = force_summary_df.sort_values('train_percent')
        force_summary_df.to_csv('force_metrics_summary.csv', index=False)
        print("Force metrics summary saved to force_metrics_summary.csv")
        
        # Create visualization of force prediction metrics
        plt.figure(figsize=(15, 12))
        
        # Plot correlation coefficient for lift
        plt.subplot(2, 2, 1)
        plt.errorbar(
            force_summary_df['train_percent'],
            force_summary_df['test_corr_lift_mean'],
            yerr=force_summary_df['test_corr_lift_std'],
            fmt='o-',
            capsize=5,
            color='blue'
        )
        plt.xlabel('Training Data Percentage')
        plt.ylabel('Correlation Coefficient')
        plt.title('Lift Prediction Correlation')
        plt.grid(True)
        
        # Plot correlation coefficient for drag
        plt.subplot(2, 2, 2)
        plt.errorbar(
            force_summary_df['train_percent'],
            force_summary_df['test_corr_drag_mean'],
            yerr=force_summary_df['test_corr_drag_std'],
            fmt='o-',
            capsize=5,
            color='red'
        )
        plt.xlabel('Training Data Percentage')
        plt.ylabel('Correlation Coefficient')
        plt.title('Drag Prediction Correlation')
        plt.grid(True)
        
        # Plot R² for lift
        plt.subplot(2, 2, 3)
        plt.errorbar(
            force_summary_df['train_percent'],
            force_summary_df['test_r2_lift_mean'],
            yerr=force_summary_df['test_r2_lift_std'],
            fmt='o-',
            capsize=5,
            color='blue'
        )
        plt.xlabel('Training Data Percentage')
        plt.ylabel('R² Score')
        plt.title('Lift Prediction R²')
        plt.grid(True)
        
        # Plot R² for drag
        plt.subplot(2, 2, 4)
        plt.errorbar(
            force_summary_df['train_percent'],
            force_summary_df['test_r2_drag_mean'],
            yerr=force_summary_df['test_r2_drag_std'],
            fmt='o-',
            capsize=5,
            color='red'
        )
        plt.xlabel('Training Data Percentage')
        plt.ylabel('R² Score')
        plt.title('Drag Prediction R²')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('force_prediction_performance.png')
        print("Force prediction performance visualization saved to force_prediction_performance.png")
        
        # Create a second visualization for MAE metrics
        plt.figure(figsize=(15, 8))
        
        # Plot MAE for lift
        plt.subplot(1, 2, 1)
        plt.errorbar(
            force_summary_df['train_percent'],
            force_summary_df['test_mae_lift_mean'],
            yerr=force_summary_df['test_mae_lift_std'],
            fmt='o-',
            capsize=5,
            color='blue'
        )
        plt.xlabel('Training Data Percentage')
        plt.ylabel('Mean Absolute Error')
        plt.title('Lift Prediction MAE')
        plt.grid(True)
        
        # Plot MAE for drag
        plt.subplot(1, 2, 2)
        plt.errorbar(
            force_summary_df['train_percent'],
            force_summary_df['test_mae_drag_mean'],
            yerr=force_summary_df['test_mae_drag_std'],
            fmt='o-',
            capsize=5,
            color='red'
        )
        plt.xlabel('Training Data Percentage')
        plt.ylabel('Mean Absolute Error')
        plt.title('Drag Prediction MAE')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('force_prediction_mae.png')
        print("Force prediction MAE visualization saved to force_prediction_mae.png")
    
    # Print checkpoint verification results
    print("\n===== Checkpoint Verification Summary =====")
    num_experiments = len(detailed_results)
    num_with_checkpoints = sum(1 for r in detailed_results if r['checkpoint_exists'])
    print(f"Total experiments analyzed: {num_experiments}")
    print(f"Experiments with valid checkpoints: {num_with_checkpoints} ({num_with_checkpoints/num_experiments*100:.1f}%)")
    print(f"Experiments missing checkpoints: {len(missing_checkpoints)} ({len(missing_checkpoints)/num_experiments*100:.1f}%)")
    
    # Save missing checkpoints information
    if missing_checkpoints:
        missing_df = pd.DataFrame(missing_checkpoints)
        missing_df.to_csv('missing_checkpoints.csv', index=False)
        print("\nDetails of missing checkpoints saved to missing_checkpoints.csv")
        
        # Count missing checkpoints by training percentage
        percent_missing = missing_df.groupby('train_percent').size()
        print("\nMissing checkpoints by training percentage:")
        for percent, count in percent_missing.items():
            print(f"  {percent}%: {count} missing")
    else:
        print("\nAll best model checkpoints are available!")
    
    # Create visualizations
    # First figure: Test metrics vs training percentage
    plt.figure(figsize=(12, 8))
    
    # Plot test RMSE vs training percentage
    plt.subplot(2, 1, 1)
    plt.errorbar(
        summary_df['train_percent'], 
        summary_df['test_rmse_mean'], 
        yerr=summary_df['test_rmse_std'],
        fmt='o-', 
        capsize=5
    )
    plt.xlabel('Training Data Percentage')
    plt.ylabel('Test RMSE')
    plt.title('Test RMSE vs Training Data Percentage')
    plt.grid(True)
    
    # Plot test MAE vs training percentage
    plt.subplot(2, 1, 2)
    plt.errorbar(
        summary_df['train_percent'], 
        summary_df['test_mae_mean'], 
        yerr=summary_df['test_mae_std'],
        fmt='o-', 
        capsize=5,
        color='green'
    )
    plt.xlabel('Training Data Percentage')
    plt.ylabel('Test MAE')
    plt.title('Test MAE vs Training Data Percentage')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('cross_validation_metrics.png')
    print("Performance metrics visualization saved to cross_validation_metrics.png")
    
    # Second figure: Best epoch vs training percentage
    plt.figure(figsize=(10, 6))
    plt.errorbar(
        summary_df['train_percent'], 
        summary_df['best_epoch_mean'], 
        yerr=summary_df['best_epoch_std'],
        fmt='o-', 
        capsize=5,
        color='purple'
    )
    
    # Add min/max range as a shaded area
    plt.fill_between(
        summary_df['train_percent'],
        summary_df['best_epoch_min'],
        summary_df['best_epoch_max'],
        color='purple',
        alpha=0.2
    )
    
    plt.xlabel('Training Data Percentage')
    plt.ylabel('Epoch of Best Validation Loss')
    plt.title('Convergence Speed vs Training Data Percentage')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('convergence_epochs.png')
    print("Convergence epoch visualization saved to convergence_epochs.png")
    
    # Create a third figure: Checkpoint availability by training percentage
    plt.figure(figsize=(10, 6))
    
    # Calculate percentage of available checkpoints by training data percentage
    if detailed_results:
        checkpoint_df = pd.DataFrame(detailed_results)
        availability = checkpoint_df.groupby('train_percent')['checkpoint_exists'].agg(
            ['count', 'sum']
        )
        availability['percentage'] = (availability['sum'] / availability['count']) * 100
        
        # Plot percentage of available checkpoints
        plt.bar(
            availability.index,
            availability['percentage'],
            color='blue',
            alpha=0.7
        )
        
        for i, percent in enumerate(availability.index):
            count = availability.loc[percent, 'sum']
            total = availability.loc[percent, 'count']
            plt.text(
                percent, 
                availability.loc[percent, 'percentage'] + 2, 
                f"{count}/{total}",
                ha='center'
            )
        
        plt.xlabel('Training Data Percentage')
        plt.ylabel('Checkpoint Availability (%)')
        plt.title('Best Model Checkpoint Availability by Training Data Size')
        plt.ylim(0, 105)  # Max 100% with some margin for text
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig('checkpoint_availability.png')
        print("Checkpoint availability visualization saved to checkpoint_availability.png")

if __name__ == "__main__":
    main()