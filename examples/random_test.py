import torch
import torch.nn as nn
import numpy as np
from functions import slope_bounds
from functions_1 import lipsdp_local_lip

# Set random seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)

# Define network dimensions and activation
dims = [2, 10, 10, 2]
activation_of_net = 'Tanh'

# Generate the network
if activation_of_net == 'Tanh':
    activation = nn.Tanh
elif activation_of_net == 'ReLU':
    activation = nn.ReLU
else:
    raise ValueError(f"Unsupported activation function: {activation_of_net}")

# Define the neural network
net = nn.Sequential(
    nn.Linear(dims[0], dims[1]),
    activation(),
    nn.Linear(dims[1], dims[2]),
    activation(),
    nn.Linear(dims[2], dims[3])
)

# Calculate the total number of neurons across hidden layers
num_neurons = sum(dims[1:-1])

# Define center and epsilon for the first computation
center = torch.zeros(dims[0], 1)
epsilon = 100 * torch.ones(dims[0], 1)

# Compute slope bounds
alpha_param, beta_param = slope_bounds(net, center, epsilon)

# Options for Lipschitz computation
options = {
    'verbose': 1,
    'solver': 'mosek'
}
mode = 'upper'

# Compute Lipschitz bound
bound, _, _ = lipsdp_local_lip(net, alpha_param, beta_param, options, mode)
print(f"Lipschitz bound (epsilon=100): {bound}")

# Redefine center and epsilon for the second computation
center = torch.zeros(dims[0], 1)
epsilon = 0.1 * torch.ones(dims[0], 1)

# Compute slope bounds again
alpha_param, beta_param = slope_bounds(net, center, epsilon)

# Compute Lipschitz bound again
bound, _, _ = lipsdp_local_lip(net, alpha_param, beta_param, options, mode)
print(f"Lipschitz bound (epsilon=0.1): {bound}")