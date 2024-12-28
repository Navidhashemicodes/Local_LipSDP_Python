# import pdb
import numpy as np
import old_code as po
import torch
from old_code import utils
from timeit import default_timer as timer


# # layer_configs = [(20, 20, 1), ]
# width = 10
# layer_configs = [(width, width, 1), ]

# # layer_configs = [(2, 5, 1), ]
# repeats = 1



def get_weights(net):

    num_layers = int((len(net)-1)/2)

    # network dimensions
    #dim_in = int(net[0].weight.shape[1])
    #dim_out = int(net[-1].weight.shape[0])
    #hidden_dims = [int(net[2*i].weight.shape[0]) for i in range(0,num_layers)]
    #dims = [dim_in] + hidden_dims + [dim_out]

    # get weights
    weights = np.zeros((num_layers+1,), dtype=object)
    weights[:] = [net[2*i].weight.detach().numpy().astype(np.float64) for i in range(0,num_layers+1)]


    # get biases
    biases = np.zeros((num_layers+1,), dtype=object)
    biases[:] = [net[2*i].bias.detach().numpy().astype(np.float64).reshape(-1,1) for i in range(0,num_layers+1)]

    return weights,biases





def compare_bounds(layer_configs,network,X,epsilon,repeats):
    """
    This is the default setting where we compute upper bounds
    for the Lipschitz constant over the hypercube [0, 1]^d.
    However, we might as well pass different upper and lower
    bounds corresponding to a different 'bounding box' using
    the variables lb (lower bound), and ub (upper bound).
    They correspond to vectors with bounds for each coordinate.
    This should be the only change in order to obtain bounds
    on Local lipschitz constant on sets of that form.
    """
#     lb = np.repeat(0., width)  # lower bounds for domain
#     ub = np.repeat(1., width)  # upper bounds for domain
    lb=X-epsilon
    ub=X+epsilon
    results = dict()
    for layer_config in layer_configs:
        res = []
        tight_s = []
        tight_p = []
        for _ in range(repeats):
            
#             network = utils.fc(layer_config)
#             weights, biases = utils.weights_from_pytorch(network)
            weights, biases = get_weights(network)
            fc = po.FullyConnected(weights, biases)
            f = fc.grad_poly
            g, lb, ub = fc.new_krivine_constr(p=1, lb=lb, ub=ub)
            start = timer()
            m = po.KrivineOptimizer.new_maximize_serial(
                    f, g, lb=lb, ub=ub, deg=len(weights),
                    start_indices=fc.start_indices,
                    layer_config=layer_config,
                    solver='gurobi', name='')
            end = timer()
            print('time elapsed: ', end - start)
            # m = po.KrivineOptimizer.maximize(
            #        f, g, deg=len(weights),
            #        solver='gurobi', n_jobs=-1, name='')
            lp_bound = m[0].objVal
            print('LP BOUND: ', lp_bound)
            ubp = po.upper_bound_product(fc.weights, p=1)
            lbp = po.lower_bound_product(fc.weights, p=1)
            print('LOWER BOUND PRODUCT: ', lbp)
            print('UPPER BOUND PRODUCT: ', ubp)
            return

            res.append(ubp / lp_bound)
            tight_p.append(lp_bound / lbp)

        results[layer_config] = {
                'lp/product': sum(tight_p) / len(tight_p),
                'product/lp': sum(res) / len(res)
                }

    return results


def main():
    # np.random.seed(4)
    seed = 7
    np.random.seed(seed)
    torch.manual_seed(seed)
    results = compare_bounds(layer_configs)
    print(results)


if __name__ == '__main__':
    main()

