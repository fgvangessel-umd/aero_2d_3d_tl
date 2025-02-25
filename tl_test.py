import torch
from tl_model import AirfoilTransformerModel
from tl_data import create_dataloaders, AirfoilDataScaler
from experiment import ExperimentManager
from tl_viz import plot_3d_wing_predictions
from utils import load_checkpoint, calculate_airfoil_forces, select_case
from config import TrainingConfig
from typing import Dict, Optional
import argparse
import numpy as np
from scipy import stats
from typing import Tuple
import sys
from matplotlib import pyplot as plt
from validation import ModelValidator

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
                    case_data = {'case_id': case_id, 'mach': mach[0].item(), 'reynolds':true_reynolds[0].item()}
                    fname = f'model_test/mach/predictions_{case_id}.png'

                    plot_3d_wing_predictions(xy_2d, xy_3d, p_2d, p_3d_true, p_3d_pred, z_coord, case_data, fname)
                    
                    '''
                    print(pressure_3d.size())
                    print(predictions.size())
                    print(geometry_3d[:,:,1:3].size())
                    
                    # Calculate forces
                    wing_lift_pred, wing_drag_pred = 0.0, 0.0
                    wing_lift_true, wing_drag_true = 0.0, 0.0
                    for i in range(xy_3d.shape[0]):
                        segment_forces_pred, total_force_pred = \
                                    calculate_airfoil_forces(xy_3d[i, ...], p_3d_pred[i,...].squeeze(), alpha=np.deg2rad(2.5))
                        segment_forces_true, total_force_true = \
                                    calculate_airfoil_forces(xy_3d[i, ...], p_3d_true[i,...].squeeze(), alpha=np.deg2rad(2.5))

                        wing_lift_pred += total_force_pred[1]
                        wing_drag_pred += total_force_pred[0]
                        wing_lift_true += total_force_true[1]
                        wing_drag_true += total_force_true[0]


                    cl_pe = (wing_lift_true-wing_lift_pred)/wing_lift_true*100
                    cd_pe = (wing_drag_true-wing_drag_pred)/wing_drag_true*100

                    print(f'Case ID: {case_id:>4}')
                    print(f"Wing Pred: Cl = {wing_lift_pred:.2f}, Cd = {wing_drag_pred:.2f}")
                    print(f"Wing True: Cl = {wing_lift_true:.2f}, Cd = {wing_drag_true:.2f}")
                    #print(f'Case ID: {case_id:>4}, MSE Error: {np.linalg.norm(p_3d_pred-p_3d_true):.2e}, Lift PE: {cl_pe: .2f}, Drag PE: {cd_pe: .2f}\n\n')

                    true_lift.append(wing_lift_true)
                    pred_lift.append(wing_lift_pred)
                    true_drag.append(wing_drag_true)
                    pred_drag.append(wing_drag_pred)
                    '''

                    force_metrics = validator.compute_force_metrics(predictions, pressure_3d, geometry_3d[:,:,1:3], alpha=2.5)
                    print(force_metrics)

                    sys.exit('DEBUG')
            

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