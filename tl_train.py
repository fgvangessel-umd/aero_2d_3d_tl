import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import sys
from tl_model import AirfoilTransformerModel
from tl_data import create_dataloaders, AirfoilDataScaler
from tl_viz import visualize_predictions
from experiment import ExperimentManager
from config import TrainingConfig
from matplotlib import pyplot as plt
import argparse

# Evaluate loss
def evaluate(model, dataloader, criterion, device, scaler):
    model.eval()
    total_loss = 0
    
    for batch in dataloader:

        # Scale batch
        batch = scaler.transform(batch)

        # Move data to device
        airfoil_2d = batch['airfoil_2d'].to(device)
        geometry_3d = batch['geometry_3d'].to(device)
        pressure_3d = batch['pressure_3d'].to(device)
        mach = batch['mach'].to(device)
        reynolds = batch['reynolds'].to(device)
        z_coord = batch['z_coord'].to(device)
        case_id = batch['case_id']
        
        # Forward pass
        predicted_pressures = model(
            airfoil_2d,
            geometry_3d,
            mach,
            reynolds,
            z_coord
        )
        
        # Compute loss
        loss = criterion(predicted_pressures, pressure_3d)
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

# Example training loop
def train_epoch(model, train_loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0
    
    for batch in train_loader:

        # Scale batch
        batch = scaler.transform(batch)

        # Move data to device
        airfoil_2d = batch['airfoil_2d'].to(device)
        geometry_3d = batch['geometry_3d'].to(device)
        pressure_3d = batch['pressure_3d'].to(device)
        mach = batch['mach'].to(device)
        reynolds = batch['reynolds'].to(device)
        z_coord = batch['z_coord'].to(device)
        case_id = batch['case_id']
        
        # Forward pass
        predicted_pressures = model(
            airfoil_2d,
            geometry_3d,
            mach,
            reynolds,
            z_coord
        )
        
        # Compute loss
        loss = criterion(predicted_pressures, pressure_3d)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def train_step(model, batch, optimizer, criterion, device, scaler):

    # Scale batch
    batch = scaler.transform(batch)

    # Move data to device
    airfoil_2d = batch['airfoil_2d'].to(device)
    geometry_3d = batch['geometry_3d'].to(device)
    pressure_3d = batch['pressure_3d'].to(device)
    mach = batch['mach'].to(device)
    reynolds = batch['reynolds'].to(device)
    z_coord = batch['z_coord'].to(device)
    case_id = batch['case_id']
    
    # Forward pass
    predicted_pressures = model(
        airfoil_2d,
        geometry_3d,
        mach,
        reynolds,
        z_coord
    )
    
    # Compute loss
    loss = criterion(predicted_pressures, pressure_3d)
    
    # Backward pass and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss

# Modified training loop
def train_model(config_path):
    # Load configuration
    config = TrainingConfig.load(config_path)
    experiment = ExperimentManager(config)

    # Create dataloaders
    dataloaders = create_dataloaders(config.data_path, batch_size=config.batch_size)
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']

    # Load or Create and fit the scaler
    scaler_fname = config.scaler_fname
    try:
        print(f"Loading scaler from {scaler_fname}")
        scaler = AirfoilDataScaler()
        scaler.load(scaler_fname)
    except FileNotFoundError:
        print(f"Fitting scaler and saving to {scaler_fname}")
        scaler = AirfoilDataScaler()
        scaler.fit(dataloaders['train'])  # Fit on training data only
        scaler.save(scaler_fname)
    
    # Your existing setup code here
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AirfoilTransformerModel(config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    criterion = torch.nn.MSELoss()
    
    # Training loop
    global_step = 0
    for epoch in range(config.num_epochs):
        model.train()
        epoch_metrics = {'train_loss': 0.0}
        
        for batch_idx, batch in enumerate(train_loader):
            # Your existing training step code here
            loss = train_step(model, batch, optimizer, criterion, device, scaler)
            
            # Log batch metrics
            experiment.log_batch_metrics({
                'train_batch_loss': loss,
                'learning_rate': optimizer.param_groups[0]['lr']
            }, global_step)
            
            global_step += 1
            epoch_metrics['train_loss'] += loss.item()
            
        # Compute epoch metrics
        epoch_metrics['train_loss'] /= len(train_loader)
        val_loss = evaluate(model, val_loader, criterion, device, scaler)
        epoch_metrics['val_loss'] = val_loss
        
        # Log epoch metrics and visualizations
        experiment.log_epoch_metrics(epoch_metrics, epoch)
        experiment.log_model_predictions(model, val_loader, device, epoch, config.num_output_figs, scaler)
        
        # Save checkpoint
        if (epoch + 1) % config.checkpoint_freq == 0:
            experiment.save_checkpoint(model, optimizer, epoch, epoch_metrics)
    
    # Finish experiment
    experiment.finish()

if __name__ == "__main__":
    # Load config file info
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    
    # Train model
    train_model(args.config)


'''
### Old

# Create dataloaders
data_path = 'data'
dataloaders = create_dataloaders(data_path, batch_size=64)


# Load or Create and fit the scaler
scaler_fname = 'airfoil_scaler.pt'
try:
    print(f"Loading scaler from {scaler_fname}")
    scaler = AirfoilDataScaler()
    scaler.load('airfoil_scaler.pt')
except FileNotFoundError:
    print(f"Fitting scaler and saving to {scaler_fname}")
    scaler = AirfoilDataScaler()
    scaler.fit(dataloaders['train'])  # Fit on training data only
    scaler.save('airfoil_scaler.pt')

# Example configuration and model instantiation
class ModelConfig:
    def __init__(self):
        self.d_model = 256        # Embedding dimension
        self.n_head = 8          # Number of attention heads
        self.n_layers = 6        # Number of decoder layers
        self.d_ff = 1024         # Feedforward network dimension
        self.dropout = 0.1       # Dropout rate

# Model Config Instance
config = ModelConfig()


# Initialize model, optimizer, and criterion
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AirfoilTransformerModel(config).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.MSELoss()

# Save path for model checkpointing
checkpoint_folder = 'trained_model/'
trained_model_fname = 'trained_model.pt'

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    train_loss = train_epoch(model, dataloaders['train'], optimizer, criterion, device, scaler.scalers)
    print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.3e}')

checkpoint = {
        'model_state_dict': model.state_dict(),
        'scaler': scaler.scalers  # Save scaler state
    }
save_path = checkpoint_folder+trained_model_fname
torch.save(checkpoint, save_path)
print(f"Model saved to {save_path}")

# Load your trained model
checkpoint = torch.load(save_path)
model.load_state_dict(checkpoint['model_state_dict'])
#scaler = AirfoilDataScaler()
#scaler.scalers = checkpoint['scaler']
print(f"Model loaded from {trained_model_fname}")
print(scaler.scalers)

train_loss = eval_loss(model, dataloaders['train'], criterion, device, scaler.scalers)
val_loss = eval_loss(model, dataloaders['val'], criterion, device, scaler.scalers)
test_loss = eval_loss(model, dataloaders['test'], criterion, device, scaler.scalers)
print(f'Train Loss: {train_loss:.3e}')
print(f'Val Loss:   {val_loss:.3e}')
print(f'Test Loss:  {test_loss:.3e}')

#visualize_predictions(model, dataloaders['train'], scaler.scalers, device, num_samples=64, save_path='predictions')
#visualize_predictions(model, dataloaders['val'], scaler.scalers, device, num_samples=64, save_path='predictions_val')
#visualize_predictions(model, dataloaders['test'], scaler.scalers, device, num_samples=64, save_path='predictions_test')
'''

