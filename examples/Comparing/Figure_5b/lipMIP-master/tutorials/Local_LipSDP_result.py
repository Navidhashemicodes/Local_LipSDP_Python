import torch
import numpy as np
import sys
import os

# Get the current directory
dir_path = os.getcwd()

# Move up the directory structure to find the Formula_factory directory
for _ in range(5):
    dir_path = os.path.dirname(dir_path)

# Add the path to the Formula_factory directory (assuming functions.py is in this directory)
functions_path = os.path.join(dir_path, 'Formula_factory')  # Adjust this if necessary
sys.path.append(functions_path)

# Now you can import the functions from functions.py
from prebound_functions import slope_bounds
from Lip_functions import lipsdp_local_lip


# Assuming Comparison.mat is loaded as a PyTorch model
# Load the PyTorch model (replace this with actual path)
net = torch.load('2by2comparison.pt')  # Replace with actual file path

# Set up the options and mode
options = {'solver': 'MOSEK', 'verbose': 1}
mode = 'upper'

# Define epsilon values using linspace
epsilons = np.linspace(0.001, 1, 10)


dimin = net[0].weight.shape[1]
X = torch.zeros(dimin, 1)

# Initialize arrays for the results
bound = np.zeros((len(epsilons),1))
time = np.zeros(( len(epsilons),1))
status = [None] * len(epsilons)

# Parallel processing using PyTorch and multiprocessing (can be adjusted based on the system)
# Iterate over each epsilon value and compute the bounds
for i in range(len(epsilons)):
    epsilon = epsilons[i] * torch.ones_like(X)
    alpha_param, beta_param = slope_bounds(net, X, epsilon)
    bound[i], time[i], status[i] = lipsdp_local_lip(net, alpha_param, beta_param, options, mode)

# Output the results
print(f"The Lipstchitz certificates, for each epsilon in {list(epsilons)}, from Local_LipSDP are: {bound.tolist()} respectively.")
