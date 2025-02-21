import torch
import numpy as np
from pathlib import Path

def calculate_lift_drag_from_pressure(x, y, p, angle_of_attack_degrees=2.5):
    """
    Calculate lift and drag coefficients from pressure distribution around airfoil sections.
    
    Parameters:
    ----------
    x : numpy.ndarray
        Array of x-coordinates with shape [n_sections, n_points]
    y : numpy.ndarray
        Array of y-coordinates with shape [n_sections, n_points]
    p : numpy.ndarray
        Array of pressure values with shape [n_sections, n_points]
    angle_of_attack_degrees : float
        Angle of attack in degrees (default: 2.5)
        
    Returns:
    -------
    tuple
        (lift_coefficients, drag_coefficients, section_forces)
    """
    # Convert angle of attack to radians
    angle_of_attack = np.radians(angle_of_attack_degrees)
    
    # Get dimensions
    n_sections = x.shape[0]
    n_points = x.shape[1]
    
    # Initialize result arrays
    lift_coefficients = np.zeros(n_sections)
    drag_coefficients = np.zeros(n_sections)
    section_forces = []
    
    # Process each cross-section
    for i in range(n_sections):
        # Get coordinates and pressure for this section
        x_section = x[i]
        y_section = y[i]
        p_section = p[i]
        
        # Calculate the surface normal vectors
        # We need to arrange points in counterclockwise order around the airfoil
        # This assumes points are already ordered properly around the airfoil
        dx = np.zeros_like(x_section)
        dy = np.zeros_like(y_section)
        
        # Calculate differentials using central differencing with periodic boundary
        dx[1:-1] = 0.5 * (x_section[2:] - x_section[:-2])
        dy[1:-1] = 0.5 * (y_section[2:] - y_section[:-2])
        
        # Handle endpoints (assuming closed curve)
        dx[0] = 0.5 * (x_section[1] - x_section[-1])
        dy[0] = 0.5 * (y_section[1] - y_section[-1])
        dx[-1] = 0.5 * (x_section[0] - x_section[-2])
        dy[-1] = 0.5 * (y_section[0] - y_section[-2])
        
        # Normal vectors (outward pointing for negative pressure)
        # Reversing dx and dy and maintaining sign gives the outward normal
        nx = -dy
        ny = dx
        
        # Normalize normal vectors
        norm = np.sqrt(nx**2 + ny**2)
        nx = nx / norm
        ny = ny / norm
        
        # Force components in x and y directions (pressure × normal × element length)
        element_length = norm
        fx = p_section * nx * element_length
        fy = p_section * ny * element_length
        
        # Sum forces to get section coefficients in airfoil coordinates
        section_force_x = np.sum(fx)
        section_force_y = np.sum(fy)
        
        # Transform to lift and drag using angle of attack
        # Lift is perpendicular to free stream, drag is parallel
        lift = -section_force_x * np.sin(angle_of_attack) + section_force_y * np.cos(angle_of_attack)
        drag = section_force_x * np.cos(angle_of_attack) + section_force_y * np.sin(angle_of_attack)
        
        # Store results
        lift_coefficients[i] = lift
        drag_coefficients[i] = drag
        section_forces.append({
            'x_force': section_force_x,
            'y_force': section_force_y,
            'lift': lift,
            'drag': drag
        })

def select_batches(tensor, batch_indices):
    """
    Select specific batches from a 3D tensor
    
    Args:
        tensor: A 3D PyTorch tensor of shape [batch_size, dim1, dim2]
        batch_indices: List or tensor of indices to select from batch dimension
        
    Returns:
        A tensor containing only the selected batches
    """
    # Convert indices to tensor if they're in a list
    if isinstance(batch_indices, list):
        batch_indices = torch.tensor(batch_indices, device=tensor.device)
        
    # Select the specified batches
    selected_batches = tensor[batch_indices]
    
    return selected_batches

def load_checkpoint(checkpoint_path, model, optimizer, scaler):
    """
    Load model checkpoint and restore scaler state
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: PyTorch model
        optimizer: PyTorch optimizer
        scaler: AirfoilDataScaler instance
    """
    # Use weights_only=False to handle numpy scalars in the checkpoint
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    
    # Update model and optimizer states
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scaler directly if it was saved as a dictionary
    if isinstance(checkpoint['scaler_state'], dict):
        scaler.scalers = checkpoint['scaler_state']
    else:
        # Create temporary file for scaler state
        temp_scaler_path = Path("temp_scaler_load.pt")
        
        # Write scaler state to temporary file
        with open(temp_scaler_path, 'wb') as f:
            f.write(checkpoint['scaler_state'])
        
        # Load scaler state
        scaler.load(temp_scaler_path)
        
        # Clean up temporary file
        if temp_scaler_path.exists():
            temp_scaler_path.unlink()
    
    return model, optimizer, scaler, checkpoint['epoch'], checkpoint['metrics']