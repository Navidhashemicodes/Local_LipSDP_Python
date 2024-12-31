import torch
import numpy as np
import cvxpy as cp
from prebound_functions import get_weights, slope_bounds
from scipy.sparse import block_diag, eye, hstack, vstack
# from scipy.linalg import block_diag





# def lipSDP(net,alpha = 0, beta =1):
    
#     num_layers = int((len(net)-1)/2)
    
#     weights = np.zeros((num_layers+1,), dtype=object)
#     weights[:] = [net[2*i].weight.detach().numpy().astype(np.float64) for i in range(0,num_layers+1)]
    
#     dim_in = weights[0].shape[1]

#     dim_last_hidden = weights[-1].shape[1]
#     hidden_dims = [weights[i].shape[0] for i in range(0,num_layers)]

#     num_neurons = sum(hidden_dims)

#     # decision vars
#     Lambda = cp.Variable((num_neurons,1),nonneg=True)
#     T = cp.diag(Lambda)
#     rho = cp.Variable((1,1),nonneg=True)

    
#     C = np.bmat([np.zeros((weights[-1].shape[0],dim_in+num_neurons-dim_last_hidden)),weights[-1]])
#     D = np.bmat([np.eye(dim_in),np.zeros((dim_in,num_neurons))])
    
#     A = weights[0]
#     for i in range(1,num_layers):
#         A = block_diag(A,weights[i])

#     A = np.bmat([A,np.zeros((A.shape[0],weights[num_layers].shape[1]))])
#     B = np.eye(num_neurons)
#     B = np.bmat([np.zeros((num_neurons,weights[0].shape[1])),B])
#     A_on_B = np.bmat([[A],[B]])

#     cons = [A_on_B.T@cp.bmat([[-2*alpha*beta*T,(alpha+beta)*T],[(alpha+beta)*T,-2*T]])@A_on_B+C.T@C-rho*D.T@D<<0]

#     prob = cp.Problem(cp.Minimize(rho), cons)

#     prob.solve(solver=cp.MOSEK)
    
#     return np.sqrt(rho.value)[0][0]


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
    lambda_var = cp.Variable((num_neurons,1))
    rho_sq = cp.Variable((1,1),nonneg=True)

    # Matrix definitions
    alpha_param = alpha_param.flatten()
    beta_param = beta_param.flatten()
    alpha_beta_diag = np.diag(alpha_param * beta_param)
    alpha_plus_beta_diag = np.diag(alpha_param + beta_param)
    Q = cp.bmat([
        [-2 * alpha_beta_diag @ cp.diag(lambda_var), alpha_plus_beta_diag @ cp.diag(lambda_var)],
        [alpha_plus_beta_diag @ cp.diag(lambda_var), -2 * cp.diag(lambda_var)]
    ])

    A = block_diag([weights for weights in weights_of_net[:-1]])
    A = hstack([A, np.zeros((A.shape[0], dim_last_hidden))])
    B = hstack([np.zeros((num_neurons, dim_in)), eye(num_neurons)])
    
    AB = vstack([A, B])
    
    Mmid = AB.T @ Q @ AB
    
    El = np.hstack([np.zeros((dim_last_hidden, dim_in + num_neurons - dim_last_hidden)), np.eye(dim_last_hidden)])
    E0 = np.hstack([np.eye(dim_in), np.zeros((dim_in, num_neurons))])
    Mout = rho_sq * (E0.T @ E0) - ((weights_of_net[-1] @ El).T @ (weights_of_net[-1] @ El))

    # Optimization problem
    constraints = []

    if mode == 'lower':
        objective = cp.Maximize(rho_sq)
        constraints.append(Mmid + Mout << 0)
    elif mode == 'upper':
        objective = cp.Minimize(rho_sq)
        constraints.append(Mmid - Mout << 0)
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
    bound = np.sqrt(rho_sq.value)[0][0]
    time = prob.solver_stats.solve_time
    status = prob.status

    # Display the result
    message = f"method: local lipsdp | solver: {options['solver']} | bound: {bound} | time: {time} | status: {status}"
    print(message)

    return bound, time, status


def robustness_bisection(net, input_data):
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
    
    dims_in = net[0].weight.shape[1]
    # dim_out = net[-1].weight.shape[0]
    # hidden_dims = [net[2*i].weight.shape[0] for i in range(0,num_layers)]
    
    output_data = net(input_data.T).detach().numpy()
    output_data = output_data.T

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
    
    options = {'solver': 'MOSEK', 'verbose': 1}
    mode = 'upper'
    
    # Bisection initialization
    epsi_lower = 0.001
    epsi_upper = 2  # Assume minimum L cannot be smaller than 0.01 * L_global
    epsilon = epsi_upper

    while epsi_upper - epsi_lower >= 0.0001:
        epsi = 0.5 * (epsi_lower + epsi_upper)
        epsi_vector = epsi * torch.ones_like(input_data)  # Sphere approximation with cube

        # Slope bounds computation
        alpha_param, beta_param = slope_bounds(net, input_data, epsi_vector)

        # Compute Lipschitz bound using Local-LipSDP with cvxpy
        bound,_,_ = lipsdp_local_lip(net, alpha_param, beta_param, options, mode)

        # Update bisection bounds
        if bound * epsi * np.sqrt(dims_in) >= rho:
            epsi_upper = epsi
        else:
            epsi_lower = epsi

        epsilon = epsi

    # Final Lipschitz constant and epsilon
    L = bound
    return L, epsilon, rho