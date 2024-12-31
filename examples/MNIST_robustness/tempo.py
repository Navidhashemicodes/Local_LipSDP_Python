import torch
import torch.nn as nn
import scipy.io
import os

def matlab_activation_to_pytorch(activation_str):
    """
    Maps MATLAB activation string to PyTorch activation functions.
    """
    if activation_str == 'relu':
        return nn.ReLU()
    elif activation_str == 'sigmoid':
        return nn.Sigmoid()
    elif activation_str == 'tanh':
        return nn.Tanh()
    else:
        raise ValueError(f"Unsupported activation type: {activation_str}")

def matlab_to_pytorch(matlab_file, output_file):
    """
    Converts a MATLAB structure representing a neural network into a PyTorch model.

    Args:
        matlab_file (str): Path to the .mat file containing the MATLAB structure.
        output_file (str): Path to save the PyTorch model as a .pt file.
    """
    # Load the MATLAB file
    mat_data = scipy.io.loadmat(matlab_file)
    

    net_struct = mat_data['net']
    
    # Extract weights, biases, and activation function
    weights = net_struct['weights'][0, 0]  # Assuming it's stored as a cell array
    biases = net_struct['biases'][0, 0]   # Assuming it's stored as a cell array
    activation = net_struct['activation'][0]  # Assuming it's stored as a string

    # Initialize a sequential PyTorch model
    layers = []
    for i in range(len(weights[0])):
        weight = torch.tensor(weights[0, i], dtype=torch.float32)
        bias = torch.tensor(biases[0, i].flatten(), dtype=torch.float32)

        # Linear layer
        linear = nn.Linear(weight.size(1), weight.size(0))
        linear.weight = nn.Parameter(weight)
        linear.bias = nn.Parameter(bias)
        layers.append(linear)

        # Add activation for all but the last layer
        if i < len(weights[0]) - 1:
            layers.append(matlab_activation_to_pytorch(activation))

    # Create the model
    model = nn.Sequential(*layers)

    # Save the model to the specified file
    torch.save(model, output_file)
    print(f"Model successfully converted and saved to {output_file}")

# Example usage
matlab_file = "Trained_model.mat"  # Path to your MATLAB file
output_file = "Trained_model.pt"   # Path to save the PyTorch model
matlab_to_pytorch(matlab_file, output_file)





# Load the .mat file
mat = scipy.io.loadmat('Input_data.mat')

# Extract the array (assuming it's stored under the key 'data')
Input = mat['Input']
# Convert the numpy array to a PyTorch tensor
Input  = Input.toarray()
Input = torch.tensor(Input)

# If you want to ensure the tensor is of the right type (float32)
Input = Input.float()
# Specify the filename
file_name = "Input_data.pt"

# Save the tensor as a .pt file
torch.save(Input, file_name)
