import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tl_model import AirfoilTransformerModel
from tl_data import create_dataloaders, AirfoilDataScaler

def visualize_predictions(model, dataloader, scaler, device, num_samples=1, save_path='predictions'):
    """
    Visualize model predictions against ground truth
    
    Args:
        model: Trained AirfoilTransformerModel
        dataloader: DataLoader containing test data
        scaler: Fitted AirfoilDataScaler
        device: torch device
        num_samples: Number of random samples to visualize
        save_path: Directory to save visualization plots
    """
    model.eval()
    
    # Get a batch of data
    batch = next(iter(dataloader))
    
    with torch.no_grad():
        # Move data to device
        airfoil_2d = batch['airfoil_2d'].to(device)
        geometry_3d = batch['geometry_3d'].to(device)
        pressure_3d_true = batch['pressure_3d'].to(device)
        mach = batch['mach'].to(device)
        reynolds = batch['reynolds'].to(device)
        z_coord = batch['z_coord'].to(device)
        case_ids = batch['case_id']
        
        # Scale reynolds number as in training
        reynolds = (reynolds - scaler['reynolds']['mean']) / scaler['reynolds']['std']
        
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
            
            plt.tight_layout()
            plt.savefig(f'{save_path}/case_{case_ids[i].item()}_z_{z_coord[i].item():.3f}.png')
            plt.close()

'''
def evaluate_model_performance(model, dataloader, scaler, device):
    """
    Evaluate model performance metrics
    
    Args:
        model: Trained AirfoilTransformerModel
        dataloader: DataLoader containing test data
        scaler: Fitted AirfoilDataScaler
        device: torch device
    
    Returns:
        dict: Dictionary containing performance metrics
    """
    model.eval()
    total_mse = 0
    total_mae = 0
    pressure_errors = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Move data to device
            airfoil_2d = batch['airfoil_2d'].to(device)
            geometry_3d = batch['geometry_3d'].to(device)
            pressure_3d_true = batch['pressure_3d'].to(device)
            mach = batch['mach'].to(device)
            reynolds = batch['reynolds'].to(device)
            z_coord = batch['z_coord'].to(device)
            
            # Scale reynolds number
            reynolds = (reynolds - scaler.scalers['reynolds']['mean']) / scaler.scalers['reynolds']['std']
            
            # Get predictions
            pressure_3d_pred = model(airfoil_2d, geometry_3d, mach, reynolds, z_coord)
            
            # Inverse transform predictions and true values
            batch_pred = {'pressure_3d': pressure_3d_pred}
            batch_true = {'pressure_3d': pressure_3d_true}
            
            unscaled_pred = scaler.inverse_transform(batch_pred)
            unscaled_true = scaler.inverse_transform(batch_true)
            
            # Calculate errors
            mse = torch.nn.functional.mse_loss(unscaled_pred['pressure_3d'], 
                                             unscaled_true['pressure_3d'])
            mae = torch.nn.functional.l1_loss(unscaled_pred['pressure_3d'], 
                                            unscaled_true['pressure_3d'])
            
            total_mse += mse.item()
            total_mae += mae.item()
            
            # Collect all errors for distribution analysis
            errors = (unscaled_pred['pressure_3d'] - unscaled_true['pressure_3d']).cpu().numpy()
            pressure_errors.extend(errors.flatten())
    
    # Calculate average metrics
    avg_mse = total_mse / len(dataloader)
    avg_mae = total_mae / len(dataloader)
    rmse = np.sqrt(avg_mse)
    
    # Calculate error distribution statistics
    error_std = np.std(pressure_errors)
    error_percentiles = np.percentile(pressure_errors, [25, 50, 75])
    
    return {
        'rmse': rmse,
        'mae': avg_mae,
        'error_std': error_std,
        'error_percentiles': error_percentiles
    }

# Usage example:
if __name__ == "__main__":
    # Load your trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = ModelConfig()  # Your model configuration
    model = AirfoilTransformerModel(config).to(device)
    model.load_state_dict(torch.load('trained_model.pt'))
    
    # Create dataloaders
    # Create dataloaders
    data_path = 'data'
    dataloaders = create_dataloaders(data_path, batch_size=32)

    print(model)
    sys.exit('DEBUG')
    
    # Load scaler
    scaler = AirfoilDataScaler()
    scaler.load('airfoil_scaler.pt')

    print(model)
    sys.exit('DEBUG')
    
    # Visualize predictions
    visualize_predictions(model, dataloaders['train'], scaler, device)
    
    # Evaluate performance
    metrics = evaluate_model_performance(model, dataloaders['test'], scaler, device)
    print("\nModel Performance Metrics:")
    print(f"RMSE: {metrics['rmse']:.6f}")
    print(f"MAE: {metrics['mae']:.6f}")
    print(f"Error Std: {metrics['error_std']:.6f}")
    print(f"Error Percentiles (25, 50, 75): {metrics['error_percentiles']}")
'''