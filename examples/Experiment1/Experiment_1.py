import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from functions import slope_bounds, lipsdp_local_lip
import torch.nn as nn

def generate_network(dims, activation):
    """
    Generate a PyTorch neural network (nn.Sequential) based on the specified dimensions and activation function.
    
    Parameters:
    - dims: List of integers representing the sizes of input, hidden, and output layers.
    - activation: String specifying the activation function ('Tanh', 'ReLU', etc.).
    
    Returns:
    - net: PyTorch nn.Sequential model.
    """
    layers = []
    num_layers = len(dims) - 1  # Number of layers in the network
    
    # Define activation function
    if activation == 'Tanh':
        activation_fn = nn.Tanh()
    elif activation == 'ReLU':
        activation_fn = nn.ReLU()
    else:
        raise ValueError(f"Unsupported activation function: {activation}")
    
    # Build the network
    for i in range(num_layers):
        layers.append(nn.Linear(dims[i], dims[i + 1]))  # Linear layer
        if i < num_layers - 1:  # Add activation after all but the last layer
            layers.append(activation_fn)
    
    # Create nn.Sequential model
    net = nn.Sequential(*layers)
    return net

def plot_areaerrorbar(data, x_axis, ax=None, options=None):
    """
    Plots data with mean and shaded area representing error bars.
    
    Parameters:
    - data: 2D NumPy array where rows are samples, and columns are data points.
    - x_axis: 1D NumPy array or list representing the x-axis values.
    - ax: Matplotlib Axes object to plot on. If None, creates a new figure and axes.
    - options: Dictionary for customization:
        - 'color_area': Color of the shaded area.
        - 'color_line': Color of the mean line.
        - 'alpha': Transparency of the shaded area.
        - 'line_width': Width of the mean line.
    """
    # Default options
    if options is None:
        options = {}
    color_area = options.get('color_area', [243 / 255, 169 / 255, 114 / 255])  # Orange theme
    color_line = options.get('color_line', [52 / 255, 148 / 255, 186 / 255])   # Blue theme
    alpha = options.get('alpha', 0.5)
    line_width = options.get('line_width', 2)

    # Compute mean and error
    data_mean = np.mean(data, axis=0)
    error = np.max(data, axis=0) - np.min(data, axis=0)

    # Prepare figure and axes
    if ax is None:
        fig, ax = plt.subplots()

    # Plot shaded area
    ax.fill_between(x_axis, data_mean + error / 2, data_mean - error / 2, 
                    color=color_area, alpha=alpha, edgecolor='none')
    
    # Plot mean line
    ax.plot(x_axis, data_mean, color=color_line, linewidth=line_width)

    # Set grid and other visual improvements
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)








# Dynamically add Formula_factory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
formula_factory_path = os.path.join(project_root, 'Formula_factory')

if formula_factory_path not in sys.path:
    sys.path.append(formula_factory_path)

# Prepare tiled layout (2x4 subplots)
fig, axes = plt.subplots(2, 4, figsize=(15, 8), tight_layout=True)
axes = axes.ravel()

# Case configurations
cases = [
    [50, 20, 10], [50, 20, 20, 10], [50, 20, 20, 20, 10], [50, 20, 20, 20, 20, 10],
    [100, 50, 10], [100, 50, 50, 10], [100, 50, 50, 50, 10], [100, 50, 50, 50, 50, 10]
]

case_results = [None] * len(cases)
case_info = [None] * len(cases)

# Common parameters
options = {'solver': 'mosek', 'verbose': 1}
mode = 'upper'
NNN = 50
epsilons = np.concatenate([np.linspace(0.01, 0.51, 26), [0.7, 0.9, 1]])

# Iterate over each case
for m, dims in enumerate(cases):
    dimin = dims[0]
    dimout = dims[-1]
    X = np.zeros((dimin, 1))

    if m > 3:
        NNN = 20

    bound = np.zeros((NNN, len(epsilons)))
    time = np.zeros((NNN, len(epsilons)))
    status = [[None] * len(epsilons) for _ in range(NNN)]
    bound_global = np.zeros(NNN)

    for ii in range(NNN):
        # Generate the network as nn.Sequential
        # Assuming a utility function `generate_network` is available
        net = generate_network(dims, activation='Tanh')

        # Parallel loop over epsilons
        for jj, epsilon_value in enumerate(epsilons):
            epsilon = epsilon_value * np.ones_like(X)
            alpha_param, beta_param = slope_bounds(net, X, epsilon)
            bound[ii, jj], time[ii, jj], status[ii][jj] = lipsdp_local_lip(net, alpha_param, beta_param, options, mode)

        # Compute global bound
        epsilon = 100 * np.ones_like(X)
        alpha_param, beta_param = slope_bounds(net, X, epsilon)
        bound_global[ii] = lipsdp_local_lip(net, alpha_param, beta_param, options, mode)

    # Normalize bounds
    case_results[m] = bound / bound_global[:, None]
    case_info[m] = (time, status)

    # Plot data
    ax = axes[m]
    plot_areaerrorbar(case_results[m], epsilons, ax=ax)

    # Format axes
    ax.set_xlim([0, 1])
    ax.set_ylim([0.6, 1.1])
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_yticks([0.6, 0.7, 0.8, 0.9, 1, 1.1])
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_ylabel('$L_{loc}/L_{global}$', fontsize=14)
    ax.text(0.5, 0.55, '$\epsilon$', fontsize=20)
    ax.text(-0.2, 1.05, f'({chr(97 + m)})', fontsize=20)
    ax.boxplot()

# Save results and plot
plt.savefig('Casestudy1.eps', format='eps', dpi=300)
plt.close(fig)

np.savez("Experiment_1.npz", cases=cases, case_results=case_results, case_info=case_info)
