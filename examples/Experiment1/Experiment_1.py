import torch
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

current_dir = os.getcwd()
formula_factory_dir = os.path.join(current_dir, "..", "..", "Formula_factory")
sys.path.append(formula_factory_dir)
from prebound_functions import slope_bounds, generate_network
from Lip_functions import lipsdp_local_lip
from concurrent.futures import ThreadPoolExecutor

def process_epsilon(jj, epsilon_value, X, net, options, mode, bound, time, status, ii):
    epsilon = epsilon_value * np.ones_like(X)
    alpha_param, beta_param = slope_bounds(net, X, epsilon)
    bound[ii, jj], time[ii, jj], status[ii][jj] = lipsdp_local_lip(net, alpha_param, beta_param, options, mode)
    
    
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
options = {'solver': 'MOSEK', 'verbose': 1}
mode = 'upper'
NNN = 50
epsilons = torch.concatenate([torch.linspace(0.01, 0.51, 26), torch.tensor([0.7, 0.9, 1])])

num_workers = 18


# Iterate over each case
for m, dims in enumerate(cases):
    dimin = dims[0]
    dimout = dims[-1]
    num_neurons = np.sum(dims[1:-1])
    X = torch.zeros((dimin, 1))

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

        # Parallelize the loop using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Use map to execute the function concurrently over epsilons
            executor.map(lambda jj: process_epsilon(jj, epsilons[jj], X, net, options, mode, bound, time, status, ii), range(len(epsilons)))

        # Compute global bound
        
        alpha_param = np.zeros((num_neurons,1))
        beta_param = np.ones((num_neurons,1))
        bound_global[ii],_,_ = lipsdp_local_lip(net, alpha_param, beta_param, options, mode)

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
    ax.boxplot(bound[ii, :])

# Save results and plot
plt.savefig('Casestudy1.eps', format='eps', dpi=300)
plt.savefig('Casestudy1.png', format='png', dpi=300)
plt.close(fig)

cases = np.array(cases, dtype=object)
case_results = np.array(case_results, dtype=object)
case_info = np.array(case_info, dtype=object)
np.savez("Experiment_1.npz", cases=cases, case_results=case_results, case_info=case_info)