import torch
from functions import slope_bounds, lipsdp_local_lip, Robustness_bisection

# Assuming Input_data and Trained_model are loaded as PyTorch tensors
# Load the PyTorch model and input data (replace these lines with actual loading code)
Input_data = torch.load('Input_data.pt')  # Replace with actual file path
Trained_model = torch.load('Trained_model.pt')  # Replace with actual file path

# Set up the options and mode
options = {'solver': 'mosek', 'verbose': 1}
mode = 'upper'

# Define the input variable (S in MATLAB)
Input = Input_data

# Compute slope bounds for the network
alpha_param, beta_param = slope_bounds(Trained_model, Input, 100 * torch.ones_like(Input))

# Compute the global Lipschitz bound
L_global = lipsdp_local_lip(Trained_model, alpha_param, beta_param, options, mode)

# Perform the robustness bisection
L, epsilon, rho = Robustness_bisection(Trained_model, Input, L_global)

# Compute the global epsilon scaled by L
epsilon_global = epsilon * L / L_global

# Save the results (using torch.save for PyTorch tensors)
torch.save({
    'L': L,
    'epsilon': epsilon,
    'L_global': L_global,
    'epsilon_global': epsilon_global,
    'rho': rho,
    'net': Trained_model,
    'Input': Input_data
}, 'Results.pt')

print("Results saved successfully.")
