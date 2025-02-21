import torch
from torch.utils.data import DataLoader
from tl_model import AirfoilTransformerModel
from tl_data import create_dataloaders, AirfoilDataScaler
from experiment import ExperimentManager
from validation import ModelValidator, ValidationMetrics
from tl_viz import plot_3d_wing_predictions
from utils import select_batches, load_checkpoint
from config import TrainingConfig
from typing import Dict, Optional
import argparse
import numpy as np
from scipy import stats
from typing import Tuple
import sys
from matplotlib import pyplot as plt

def calculate_airfoil_forces(
    points: np.ndarray,
    pressures: np.ndarray,
    alpha: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate nodal and total forces on an airfoil given discrete points and pressure values.
    
    Parameters
    ----------
    points : np.ndarray
        Nx2 array of (x,y) coordinates along the airfoil surface
    pressures : np.ndarray
        Length N array of pressure values at each point
    alpha : float
        Angle of attack of airfoil
        
    Returns
    -------
    nodal_forces : np.ndarray
        Nx2 array of force vectors at each node
    total_force : np.ndarray
        2-element array containing the total force vector [Fx, Fy]
        
    Notes
    -----
    Forces are calculated using a piecewise linear approximation between points.
    The direction of the force is determined by the local normal vector.
    """
    
    # Input validation
    if points.shape[0] != pressures.shape[0]:
        raise ValueError("Number of points must match number of pressure values")
    if points.shape[1] != 2:
        raise ValueError("Points must be 2D coordinates")
        
    N = points.shape[0]
    
    # Calculate vectors between adjacent points
    # Use periodic boundary for last point
    delta_points = np.roll(points, -1, axis=0) - points
    
    # Calculate length of each segment
    segment_lengths = np.sqrt(np.sum(delta_points**2, axis=1))

    # Calculate normal vectors (rotate tangent vector 90 degrees counterclockwise)
    normal_vectors = np.zeros_like(points)
    normal_vectors[:, 0] = -delta_points[:, 1] / segment_lengths  # Fixed broadcasting
    normal_vectors[:, 1] = delta_points[:, 0] / segment_lengths   # Fixed broadcasting
    
    # Calculate average pressure for each segment
    # Average between current and next point
    segment_pressures = 0.5 * (pressures + np.roll(pressures, -1))
    
    # Calculate segment forces
    # Force = pressure * length * normal_vector
    segment_forces = normal_vectors * segment_pressures[:, np.newaxis] * segment_lengths[:, np.newaxis]

    # Define rotation array to transform from airfoil affixed coordinate system into free-stream coordinate system
    R = np.array([[np.cos(alpha), np.sin(alpha)], [-np.sin(alpha), np.cos(alpha)]])
    segment_forces = segment_forces @ R.T
    
    # Calculate total force by summing all segment forces
    total_force = np.sum(segment_forces, axis=0)
    
    return segment_forces, total_force

if __name__ == "__main__":
    """Main testing function"""
    # Load config file info
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()

    # Load configuration
    config = TrainingConfig.load(args.config)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dataloaders
    data_path = 'data_mach'
    dataloaders = create_dataloaders(
        data_path,
        batch_size=10000,
        num_workers=config.num_workers
    )
    
    # Initialize or load scaler
    scaler = AirfoilDataScaler()
    
    # Initialize model and training components
    model = AirfoilTransformerModel(config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    criterion = torch.nn.MSELoss()

    # Set checkpoint path
    checkpoint_path = 'experiments/baseline_transformer_mach_20250218_152538/models/checkpoint_epoch_605.pt'

    # Load checkpointed model
    model, optimizer, scaler, epoch, metrics = load_checkpoint(checkpoint_path, model, optimizer, scaler)

    model.eval()

    with torch.no_grad():

        # Get val cases
        for batch_idx, batch in enumerate(dataloaders['val']):
            # Move data to device
            val_cases = batch['case_id'].numpy().tolist()
        val_cases = list(set(val_cases))

        # Get test cases
        for batch_idx, batch in enumerate(dataloaders['test']):
            # Move data to device
            test_cases = batch['case_id'].numpy().tolist()
        test_cases = list(set(test_cases))

        case_ids = {'val': val_cases, 'test': test_cases}
        true_lift = []
        pred_lift = []
        true_drag = []
        pred_drag = []

        # Load data
        for split in ['test']:
            pred_lifts, pred_drags = [], []
            true_lifts, true_drags = [], []
            for case_id in case_ids[split]:
                for batch_idx, batch in enumerate(dataloaders[split]):
                
                    true_reynolds = batch['reynolds'].to(device)
                    # Scale batch if scaler is provided
                    batch = scaler.transform(batch)

                        
                    # Move data to device
                    airfoil_2d = batch['airfoil_2d'].to(device)
                    geometry_3d = batch['geometry_3d'].to(device)
                    pressure_3d = batch['pressure_3d'].to(device)
                    mach = batch['mach'].to(device)
                    reynolds = batch['reynolds'].to(device)
                    z_coord = batch['z_coord'].to(device)
                    cases = batch['case_id'].numpy().tolist()

                    # Select data corresponding to a single wing
                    idxs = [i for i, x in enumerate(cases) if x == case_id]
                    try:
                        airfoil_2d = select_batches(airfoil_2d, idxs)
                        geometry_3d = select_batches(geometry_3d, idxs)
                        pressure_3d = select_batches(pressure_3d, idxs)
                        mach = select_batches(mach, idxs)
                        reynolds = select_batches(reynolds, idxs)
                        true_reynolds = select_batches(true_reynolds, idxs)
                        z_coord = select_batches(z_coord, idxs)
                    except IndexError:
                        continue
                    
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
                    case_data = {'case_id': case_id, 'mach': mach[0].item(), 'reynolds':true_reynolds[0].item()}
                    fname = f'model_test/mach/predictions_{case_id}.png'

                    plot_3d_wing_predictions(xy_2d, xy_3d, p_2d, p_3d_true, p_3d_pred, z_coord, case_data, fname)
                    
                    # Calculate forces
                    wing_lift_pred, wing_drag_pred = 0.0, 0.0
                    wing_lift_true, wing_drag_true = 0.0, 0.0
                    for i in range(xy_3d.shape[0]):
                        segment_forces_pred, total_force_pred = \
                                    calculate_airfoil_forces(xy_3d[i, ...], p_3d_pred[i,...].squeeze(), alpha=np.deg2rad(2.5))
                        segment_forces_true, total_force_true = \
                                    calculate_airfoil_forces(xy_3d[i, ...], p_3d_true[i,...].squeeze(), alpha=np.deg2rad(2.5))

                        wing_lift_pred += total_force_pred[1]/xy_3d.shape[0]
                        wing_drag_pred += total_force_pred[0]/xy_3d.shape[0]
                        wing_lift_true += total_force_true[1]/xy_3d.shape[0]
                        wing_drag_true += total_force_true[0]/xy_3d.shape[0]

                        # Print results
                        #print(f"Total force pred: Fx = {total_force_pred[0]:.2f}, Fy = {total_force_pred[1]:.2f}")
                        #print(f"Total force true: Fx = {total_force_true[0]:.2f}, Fy = {total_force_true[1]:.2f}\n")
                    cl_pe = (wing_lift_true-wing_lift_pred)/wing_lift_true*100
                    cd_pe = (wing_drag_true-wing_drag_pred)/wing_drag_true*100

                    print(f'Case ID: {case_id:>4}')
                    print(f"Wing Pred: Cl = {wing_lift_pred:.2f}, Cd = {wing_drag_pred:.2f}")
                    print(f"Wing True: Cl = {wing_lift_true:.2f}, Cd = {wing_drag_true:.2f}")
                    print(f'Case ID: {case_id:>4}, MSE Error: {np.linalg.norm(p_3d_pred-p_3d_true):.2e}, Lift PE: {cl_pe: .2f}, Drag PE: {cd_pe: .2f}\n\n')

                    true_lift.append(wing_lift_true)
                    pred_lift.append(wing_lift_pred)
                    true_drag.append(wing_drag_true)
                    pred_drag.append(wing_drag_pred)
            

            fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20,10), tight_layout=True)
            axs[0].scatter(true_lift, pred_lift)
            axs[1].scatter(true_drag, pred_drag)
            axs[0].plot(np.linspace(np.min(true_lift), np.max(true_lift)), np.linspace(np.min(true_lift), np.max(true_lift)), c='k')
            axs[1].plot(np.linspace(np.min(true_drag), np.max(true_drag)), np.linspace(np.min(true_drag), np.max(true_drag)), c='k')
            plt.savefig('lift_drag_parity.png')

            corr_lift, _ = stats.pearsonr(true_lift, pred_lift)
            corr_drag, _ = stats.pearsonr(true_drag, pred_drag)

            print(f'Correlation Lift: {corr_lift: .2e}')
            print(f'Correlation Drag: {corr_drag: .2e}')