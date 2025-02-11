import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import sys
from tl_model import AirfoilTransformerModel
from tl_data import create_dataloaders, AirfoilDataScaler
from tl_viz import visualize_predictions
from matplotlib import pyplot as plt

# Evaluate loss
def eval_loss(model, dataloader, criterion, device, scaler):
    model.eval()
    total_loss = 0
    
    for batch in dataloader:

        batch['reynolds'] = (batch['reynolds'] - scaler['reynolds']['mean'])/scaler['reynolds']['std']

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

        batch['reynolds'] = (batch['reynolds'] - scaler['reynolds']['mean'])/scaler['reynolds']['std']

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

