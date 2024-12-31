import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import fsolve
import scipy.io



def slope_bounds(net, X, epsilon):
    """
    Computes the slope bounds of a neural network over an L-infinity ball. we assume net[1] is always the activation function
    and the net's activation functions are all of the same type.

    Parameters:
        net (nn.Sequential): The neural network model.
        X (torch.Tensor): The center of the L-infinity ball.
        epsilon (float): The radius of the L-infinity ball.

    Returns:
        tuple: alpha_param, beta_param representing slope parameters.
    """
    # Identify the activation function from the network
    activation_name  = net[1].__class__.__name__
    
    # Handle cases based on the activation type
    if activation_name == 'ReLU':
        l, u = piecewise_linear_bounds_ReLU(net, X, epsilon)
        alpha_param, beta_param = piecewise_linear_slopes_ReLU(l, u)
    elif activation_name == 'Tanh' or activation_name == 'Sigmoid':
        func = lambda x: net[1](torch.tensor(x, dtype=torch.float32)).numpy() if isinstance(x, (np.ndarray, np.number)) else net[1](x).numpy()
        l, u = piecewise_linear_bounds_noReLU(net, X, epsilon, func, activation_name)
        alpha_param, beta_param = piecewise_linear_slopes_noReLU(l, u, func, activation_name)
    else:
        raise ValueError("Unsupported or missing activation function in the network.")
    
    return alpha_param, beta_param


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


def first_pre_activate(W,b,X,epsilon):
    l_first=-np.dot(np.abs(W),epsilon)+np.dot(W,X)+b
    u_first=np.dot(np.abs(W),epsilon)+np.dot(W,X)+b
        
    return l_first, u_first


###################################  noReLU part   ##########################################


def diffi(x,func, activation_name):
    if activation_name == 'Tanh':
        return  1 - func(x)**2
    elif activation_name == 'Sigmoid':
        return func(x)*(1-func(x))
    else:
        raise ValueError("Unsupported or missing activation function in the network.")
        

def d_find(d,*data):
    l,funci, activation_name = data
    S=((funci(d)-funci(l))/(d-l))-diffi(d,funci, activation_name)       
    return S
    
        
    
    
def next_layer_prebound_noReLU(W,b,l_pre,u_pre,func, activation_name):
    leng=len(l_pre)
    alph_upp=np.zeros((leng,1),float)
    alph_low=np.zeros((leng,1),float)
    alphabet_upp=np.zeros((leng,1),float)
    alphabet_low=np.zeros((leng,1),float)
    
    for i in range(0,leng):
        if l_pre[i]>0:
            d=0.5*(l_pre[i]+u_pre[i])
            alph_upp[i]=diffi(d,func, activation_name)
            alphabet_upp[i]=func(d)-alph_upp[i]*d
            ss=(func(u_pre[i])-func(l_pre[i]))/(u_pre[i]-l_pre[i])
            alph_low[i]=ss
            alphabet_low[i]=func(l_pre[i])-alph_low[i]*l_pre[i]
        elif u_pre[i]<0:
            ss=(func(u_pre[i])-func(l_pre[i]))/(u_pre[i]-l_pre[i])
            alph_upp[i]=ss
            alphabet_upp[i]=func(l_pre[i])-alph_upp[i]*l_pre[i]
                
            d=0.5*(l_pre[i]+u_pre[i])
            alph_low[i]=diffi(d,func, activation_name)
            alphabet_low[i]=func(d)-alph_low[i]*d
        else:
            data=(l_pre[i],func, activation_name)
            d=fsolve(d_find,0.01, args=data)
            if d<0 or d>u_pre[i]:
                ss=(func(u_pre[i])-func(l_pre[i]))/(u_pre[i]-l_pre[i])
                alph_upp[i]=ss
                alphabet_upp[i]=func(l_pre[i])-alph_upp[i]*l_pre[i]
            else:
                alph_upp[i]=diffi(d,func, activation_name)
                alphabet_upp[i]=func(l_pre[i])-alph_upp[i]*l_pre[i]
                 
                    
            data=(u_pre[i],func, activation_name)
            d=fsolve(d_find,0.01, args=data)
            if d>0 or d<l_pre[i]:
                ss=(func(u_pre[i])-func(l_pre[i]))/(u_pre[i]-l_pre[i])
                alph_low[i]=ss
                alphabet_low[i]=func(u_pre[i])-alph_low[i]*u_pre[i]
            else:
                alph_low[i]=diffi(d,func, activation_name)
                alphabet_low[i]=func(u_pre[i])-alph_low[i]*u_pre[i]
                
    W_pos=0.5*(W+np.abs(W))
    W_neg=0.5*(W-np.abs(W))
    mid=0.5*(l_pre+u_pre)
    dif=0.5*(u_pre-l_pre)
    
    cc1=W_pos*alph_low.T+W_neg*alph_upp.T
    dd1=np.dot(W_pos,alphabet_low) + np.dot(W_neg,alphabet_upp) + b
    l_next=-np.dot(np.abs(cc1),dif) + np.dot(cc1,mid) + dd1
    
    
    
    cc2=W_neg*alph_low.T+W_pos*alph_upp.T
    dd2=np.dot(W_neg,alphabet_low) + np.dot(W_pos,alphabet_upp) + b
    u_next=np.dot(np.abs(cc2),dif)+np.dot(cc2,mid)+dd2
    
    return l_next, u_next
    
    
def piecewise_linear_bounds_noReLU(net,X,epsilon, func, activation_name):
    
    
    num_layers = int((len(net)-1)/2)
    W,b=get_weights(net)
    ll,uu=first_pre_activate(W[0],b[0],X,epsilon)
    leng=len(ll)
    N=0
    for i in range(0,num_layers):
        N+=np.size(W[i],0)
    
    l=np.zeros((N,1),float)
    u=np.zeros((N,1),float)
    ii=np.r_[0:leng]
    l[ii]=ll
    u[ii]=uu
    num=leng-1
    for i in range(1,num_layers):
        ll,uu=next_layer_prebound_noReLU(W[i],b[i],ll,uu,func, activation_name)
        leng=len(ll)
        ii=np.r_[num+1:num+1+leng]
        num+=leng
        l[ii]=ll
        u[ii]=uu
    return l,u
    

def piecewise_linear_slopes_noReLU(l, u, func, activation_name):
    
    
    diffil=diffi(l,func, activation_name)
    diffiu=diffi(u,func, activation_name)
    diffi_1=diffi(np.zeros(l.shape,float),func, activation_name)
    diffi_2=diffi(0.5*(l+u),func, activation_name)
    diffi0= ((l*u)>0)*diffi_2 + ((l*u)<=0)*diffi_1
    alpha_param=np.minimum(diffil,diffiu)
    beta_param=np.maximum(diffi0, np.maximum(diffil,diffiu))
    
    return alpha_param,beta_param




##############################   ReLU part #########################################


##################   MIT Paper:

    
def next_layer_prebound_ReLU(W,b,l_pre,u_pre):
    leng=len(l_pre)
    alph_upp=np.zeros((leng,1),float)
    alph_low=np.zeros((leng,1),float)
    alphabet_upp=np.zeros((leng,1),float)
    alphabet_low=np.zeros((leng,1),float)
    
    for i in range(0,leng):
        if l_pre[i]>0:
            alph_upp[i]=1.0
            alphabet_upp[i]=0.0
            alph_low[i]=1.0
            alphabet_low[i]=0.0
        elif u_pre[i]<0:
            alph_upp[i]=0.0
            alphabet_upp[i]=0.0
                
            alph_low[i]=0.0
            alphabet_low[i]=0.0
        else:
            alph_upp[i]=u_pre[i]/(u_pre[i]-l_pre[i])
            alphabet_upp[i]=-l_pre[i]*u_pre[i]/(u_pre[i]-l_pre[i])

            if  u_pre[i]>abs(l_pre[i]):
                alph_low[i]=1.0
                alphabet_low[i]=0.0
            else:
                alph_low[i]=0.0
                alphabet_low[i]=0.0
                
    W_pos=0.5*(W+np.abs(W))
    W_neg=0.5*(W-np.abs(W))
    mid=0.5*(l_pre+u_pre)
    dif=0.5*(u_pre-l_pre)
    
    cc1=W_pos*alph_low.T+W_neg*alph_upp.T
    dd1=np.dot(W_pos,alphabet_low) + np.dot(W_neg,alphabet_upp) + b
    l_next=-np.dot(np.abs(cc1),dif) + np.dot(cc1,mid) + dd1
    
    
    
    cc2=W_neg*alph_low.T+W_pos*alph_upp.T
    dd2=np.dot(W_neg,alphabet_low) + np.dot(W_pos,alphabet_upp) + b
    u_next=np.dot(np.abs(cc2),dif)+np.dot(cc2,mid)+dd2
    
        
    return l_next, u_next
    
    

    
    
def piecewise_linear_bounds_ReLU(net,X,epsilon):
    num_layers = int((len(net)-1)/2)
    W,b=get_weights(net)
    ll,uu=first_pre_activate(W[0],b[0],X,epsilon)
    leng=len(ll)
    N=0
    for i in range(0,num_layers):
        N+=np.size(W[i],0)
    
    l=np.zeros((N,1),float)
    u=np.zeros((N,1),float)
    ii=np.r_[0:leng]
    l[ii]=ll
    u[ii]=uu
    num=leng-1
    for i in range(1,num_layers):
        ll,uu=next_layer_prebound_ReLU(W[i],b[i],ll,uu)
        leng=len(ll)
        ii=np.r_[num+1:num+1+leng]
        num+=leng
        l[ii]=ll
        u[ii]=uu
    return l,u


def piecewise_linear_slopes_ReLU(l,u):
    
    alpha_param= (l>0)*np.ones(l.shape,float)
    beta_param=(u>=0)*np.ones(u.shape,float)
    
    return alpha_param,beta_param




def export2matlab(file_name,net,save_model=False):
    '''
    Export pytorch fully connected network to matlab
    '''

    num_layers = int((len(net)-1)/2)
    dim_in = float(net[0].weight.shape[1])
    dim_out = float(net[-1].weight.shape[0])
    hidden_dims = [float(net[2*i].weight.shape[0]) for i in range(0,num_layers)]

    # network dimensions
    dims = [dim_in] + hidden_dims + [dim_out]

    # get weights
    weights = np.zeros((num_layers+1,), dtype=object)
    weights[:] = [net[2*i].weight.detach().numpy().astype(np.float64) for i in range(0,num_layers+1)]


    # get biases
    biases = np.zeros((num_layers+1,), dtype=object)
    biases[:] = [net[2*i].bias.detach().numpy().astype(np.float64).reshape(-1,1) for i in range(0,num_layers+1)]

    activation = str(net[1])[0:-2].lower()

    # export network data to matlab
    data = {}
    data['net'] = {'weights': weights,'biases':biases, 'dims': dims, 'activation': activation, 'name': file_name}

    scipy.io.savemat(file_name + '.mat', data)

    if save_model:
        torch.save(net, file_name + '.pt')
        

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
        