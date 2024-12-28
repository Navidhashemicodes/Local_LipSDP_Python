import numpy as np
import cvxpy as cp
from function1 import get_weights
from scipy.sparse import block_diag, eye, hstack



def lipsdp_local_lip(net, alpha_param, beta_param, options, mode):
    """
    Python implementation of the MATLAB lipsdp_local_lip function.
    
    Parameters:
        net: PyTorch model (nn.Sequential).
        alpha_param: Parameter alpha (numpy array).
        beta_param: Parameter beta (numpy array).
        options: Dictionary with solver and verbose keys.
        mode: Mode of optimization ('lower' or 'upper').

    Returns:
        bound: The calculated bound.
        time: Solver execution time.
        status: Solver status.
    """
    # Extract weights and biases from net
    weights_of_net, biases_of_net = get_weights(net)
    activation_of_net = net[1].__class__.__name__

    # Combine biases except the last one
    bb = np.vstack([b for b in biases_of_net[:-1]])

    # Input dimension
    dim_in = weights_of_net[0].shape[1]

    # Dimension of the last hidden layer
    dim_last_hidden = weights_of_net[-1].shape[1]

    # Total number of neurons
    num_neurons = bb.shape[0]

    # Set up the optimization problem
    if options['verbose']:
        print("Starting optimization...")

    # Define variables
    lambda_var = cp.Variable(num_neurons)
    rho_sq = cp.Variable(nonneg=True)

    # Matrix definitions
    alpha_beta_diag = np.diag(alpha_param * beta_param)
    alpha_plus_beta_diag = np.diag(alpha_param + beta_param)
    Q = cp.bmat([
        [-2 * alpha_beta_diag @ cp.diag(lambda_var), alpha_plus_beta_diag @ cp.diag(lambda_var)],
        [alpha_plus_beta_diag @ cp.diag(lambda_var), -2 * cp.diag(lambda_var)]
    ])

    A = block_diag([weights for weights in weights_of_net[:-1]])
    A = hstack([A, np.zeros((A.shape[0], dim_last_hidden))])
    B = hstack([np.zeros((num_neurons, dim_in)), eye(num_neurons)])

    Mmid = (A.T @ Q @ A) + (B.T @ Q @ B)
    El = np.hstack([np.zeros((dim_last_hidden, dim_in + num_neurons - dim_last_hidden)), eye(dim_last_hidden)])
    E0 = np.hstack([eye(dim_in), np.zeros((dim_in, num_neurons))])
    Mout = rho_sq * (E0.T @ E0) - ((weights_of_net[-1] @ El).T @ (weights_of_net[-1] @ El))

    # Optimization problem
    constraints = []

    if mode == 'lower':
        objective = cp.Maximize(rho_sq)
        constraints.append(Mmid + Mout <= 0)
    elif mode == 'upper':
        objective = cp.Minimize(rho_sq)
        constraints.append(Mmid - Mout <= 0)
    else:
        raise ValueError("Mode must be 'lower' or 'upper'.")

    # Add activation-specific constraints
    if activation_of_net == 'ReLU':
        ipn_indices = np.intersect1d(np.where(alpha_param == 0)[0], np.where(beta_param == 1)[0])
        for i in ipn_indices:
            constraints.append(lambda_var[i] >= 0)
    else:
        constraints.append(lambda_var >= 0)

    # Solve the problem
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=options['solver'], verbose=options['verbose'])

    # Collect results
    bound = np.sqrt(rho_sq.value)
    time = prob.solver_stats.solve_time
    status = prob.status

    # Display the result
    message = f"method: local lipsdp | solver: {options['solver']} | bound: {bound} | time: {time} | status: {status}"
    print(message)

    return bound, time, status


def robustness_bisection(net, input_data, L_g):
    """
    Epsilon finder function in Python.

    Parameters:
        net: The neural network object, with methods `eval` and `slope_bounds`.
        input_data: Input data point to evaluate.
        L_g: Global Lipschitz constant.

    Returns:
        L: Local Lipschitz constant.
        epsilon: Maximum perturbation radius.
        rho: Minimum margin between the correct and incorrect outputs.
    """

    dims_in = net.dims[0]
    output_data = net.eval(input_data)

    # Display classification
    i = np.argmax(output_data)
    print(f"The introduced data point is classified to be {i}")

    # Compute rho (minimum margin)
    leng = len(output_data)
    I = np.eye(leng)
    Rho = [
        abs((I[:, i] - I[:, j]).T @ output_data / np.sqrt(2))
        for j in range(leng) if j != i
    ]
    rho = min(Rho)

    # Bisection initialization
    epsi_lower = 0.001
    epsi_upper = 2  # Assume minimum L cannot be smaller than 0.01 * L_global
    epsilon = epsi_upper

    while epsi_upper - epsi_lower >= 0.0001:
        epsi = 0.5 * (epsi_lower + epsi_upper)
        epsi_vector = epsi * np.ones_like(input_data)  # Sphere approximation with cube

        # Slope bounds computation
        alpha_param, beta_param = net.slope_bounds(input_data, epsi_vector)

        # Compute Lipschitz bound using Local-LipSDP with cvxpy
        bound = lipsdp_local_lip(net, alpha_param, beta_param, mode="upper")

        # Update bisection bounds
        if bound * epsi * np.sqrt(dims_in) >= rho:
            epsi_upper = epsi
        else:
            epsi_lower = epsi

        epsilon = epsi

    # Final Lipschitz constant and epsilon
    L = bound
    return L, epsilon, rho