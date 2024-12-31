import torch
import os
import sys


current_dir = os.getcwd()
formula_factory_dir = os.path.join(current_dir, "..", "..", "Formula_factory")
sys.path.append(formula_factory_dir)
from Lip_functions import lipsdp_local_lip, robustness_bisection


import numpy as np


# Load the PyTorch model and input data (replace these lines with actual loading code)
Input_data = torch.load('Input_data.pt')  # Replace with actual file path
net = torch.load('Trained_model.pt')  # Replace with actual file path

# Set up the options and mode
options = {'solver': 'MOSEK', 'verbose': 1}
mode = 'upper'

# Define the input variable (S in MATLAB)
Input = Input_data

# Compute slope bounds for the network

num_layers = int((len(net)-1)/2)
num_neurons = np.sum( [net[2*i].weight.shape[0] for i in range(0,num_layers)] )


# Compute the global Lipschitz bound
alpha_param = np.zeros((num_neurons,1))
beta_param = np.ones((num_neurons,1))
L_global,_,_ = lipsdp_local_lip(net, alpha_param, beta_param, options, mode)

# Perform the robustness bisection
L, epsilon, rho = robustness_bisection(net, Input)

# Compute the global epsilon scaled by L
epsilon_global = epsilon * L / L_global

# Save the results (using torch.save for PyTorch tensors)
torch.save({
    'L': L,
    'epsilon': epsilon,
    'L_global': L_global,
    'epsilon_global': epsilon_global,
    'rho': rho,
    'net': net,
    'Input': Input_data
}, 'Results.pt')

print("Results saved successfully.")
