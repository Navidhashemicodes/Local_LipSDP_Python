import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import fsolve


def slope_bound(net, X, epsilon):
    """
    Computes the slope bounds of a neural network over an L-infinity ball.

    Parameters:
        net (nn.Sequential): The neural network model.
        X (torch.Tensor): The center of the L-infinity ball.
        epsilon (float): The radius of the L-infinity ball.

    Returns:
        tuple: alpha_param, beta_param representing slope parameters.
    """
    # Identify the activation function from the network
    activation = None
    for layer in net:
        if isinstance(layer, nn.ReLU):
            activation = "ReLU"
            break
        elif isinstance(layer, (nn.Tanh, nn.Sigmoid, nn.LeakyReLU)):  # Extend as needed
            activation = "Non-ReLU"
            break
    
    # Handle cases based on the activation type
    if activation == "ReLU":
        l, u = piecewise_linear_bounds_ReLU(net, X, epsilon)
        alpha_param, beta_param = piecewise_linear_slopes_ReLU(l, u)
    elif activation == "Non-ReLU":
        l, u = piecewise_linear_bounds_noReLU(net, X, epsilon)
        alpha_param, beta_param = piecewise_linear_slopes_noReLU(l, u)
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
    weights = np.zeros((num_layers+1,), dtype=np.object)
    weights[:] = [net[2*i].weight.detach().numpy().astype(np.float64) for i in range(0,num_layers+1)]


    # get biases
    biases = np.zeros((num_layers+1,), dtype=np.object)
    biases[:] = [net[2*i].bias.detach().numpy().astype(np.float64).reshape(-1,1) for i in range(0,num_layers+1)]

    return weights,biases


def first_pre_activate(W,b,X,epsilon):
    l_first=-np.dot(np.abs(W),epsilon)+np.dot(W,X)+b
    u_first=np.dot(np.abs(W),epsilon)+np.dot(W,X)+b
        
    return l_first, u_first


###################################  noReLU part   ##########################################

    
def diffi(x,func):
    epsi=0.00001
    y_prim= (func(x+epsi)-func(x-epsi))/(2*epsi)
    return y_prim

def d_find_upp(d,*data):
    l,funci=data
    S=((funci(d)-funci(l))/(d-l))-diffi(d,funci)       
    return S

def d_find_low(d,*data):
    u,funci=data
    S=((funci(d)-funci(u))/(d-u))-diffi(d,funci)
    return S
    
        
    
    
def next_layer_prebound_noReLU(W,b,l_pre,u_pre,func):
    leng=len(l_pre)
    alph_upp=np.zeros((leng,1),np.float)
    alph_low=np.zeros((leng,1),np.float)
    alphabet_upp=np.zeros((leng,1),np.float)
    alphabet_low=np.zeros((leng,1),np.float)
    
    for i in range(0,leng):
        if l_pre[i]>0:
            d=0.5*(l_pre[i]+u_pre[i])
            alph_upp[i]=diffi(d,func)
            alphabet_upp[i]=func(d)-alph_upp[i]*d
            ss=(func(u_pre[i])-func(l_pre[i]))/(u_pre[i]-l_pre[i])
            alph_low[i]=ss
            alphabet_low[i]=func(l_pre[i])-alph_low[i]*l_pre[i]
        elif u_pre[i]<0:
            ss=(func(u_pre[i])-func(l_pre[i]))/(u_pre[i]-l_pre[i])
            alph_upp[i]=ss
            alphabet_upp[i]=func(l_pre[i])-alph_upp[i]*l_pre[i]
                
            d=0.5*(l_pre[i]+u_pre[i])
            alph_low[i]=diffi(d,func)
            alphabet_low[i]=func(d)-alph_low[i]*d
        else:
            data=(l_pre[i],func)
            d=fsolve(d_find_upp,0.01, args=data)
            if d<0 or d>u_pre[i]:
                ss=(func(u_pre[i])-func(l_pre[i]))/(u_pre[i]-l_pre[i])
                alph_upp[i]=ss
                alphabet_upp[i]=func(l_pre[i])-alph_upp[i]*l_pre[i]
            else:
                alph_upp[i]=diffi(d,func)
                alphabet_upp[i]=func(l_pre[i])-alph_upp[i]*l_pre[i]
                 
                    
            data=(u_pre[i],func)
            d=fsolve(d_find_low,0.01, args=data)
            if d>0 or d<l_pre[i]:
                ss=(func(u_pre[i])-func(l_pre[i]))/(u_pre[i]-l_pre[i])
                alph_low[i]=ss
                alphabet_low[i]=func(u_pre[i])-alph_low[i]*u_pre[i]
            else:
                alph_low[i]=diffi(d,func)
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
    
    
def piecewise_linear_bounds_noReLU(net,X,epsilon):
    
    func = lambda x: net[1](torch.tensor(x, dtype=torch.float32)) if isinstance(x, (np.ndarray, np.number)) else net[1](x)
    
    num_layers = int((len(net)-1)/2)
    W,b=get_weights(net)
    ll,uu=first_pre_activate(W[0],b[0],X,epsilon)
    leng=len(ll)
    N=0
    for i in range(0,num_layers):
        N+=np.size(W[i],0)
    
    l=np.zeros((N,1),np.float)
    u=np.zeros((N,1),np.float)
    ii=np.r_[0:leng]
    l[ii]=ll
    u[ii]=uu
    num=leng-1
    for i in range(1,num_layers):
        ll,uu=next_layer_prebound_noReLU(W[i],b[i],ll,uu,func)
        leng=len(ll)
        ii=np.r_[num+1:num+1+leng]
        num+=leng
        l[ii]=ll
        u[ii]=uu
    return l,u
    

def piecewise_linear_slopes_noReLU(l,u,func):
    
    diffil=diffi(l,func)
    diffiu=diffi(u,func)
    diffi_1=diffi(np.zeros(l.shape,np.float),func)
    diffi_2=diffi(0.5*(l+u),func)
    diffi0= ((l*u)>0)*diffi_2 + ((l*u)<=0)*diffi_1
    alpha_param=np.minimum(diffil,diffiu)
    beta_param=np.maximum(diffi0, np.maximum(diffil,diffiu))
    
    return alpha_param,beta_param




##############################   ReLU part #########################################


##################   MIT Paper:

    
def next_layer_prebound_ReLU(W,b,l_pre,u_pre):
    leng=len(l_pre)
    alph_upp=np.zeros((leng,1),np.float)
    alph_low=np.zeros((leng,1),np.float)
    alphabet_upp=np.zeros((leng,1),np.float)
    alphabet_low=np.zeros((leng,1),np.float)
    
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
    
    l=np.zeros((N,1),np.float)
    u=np.zeros((N,1),np.float)
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
    
    alpha_param= (l>0)*np.ones(l.shape,np.float)
    beta_param=(u>=0)*np.ones(u.shape,np.float)
    
    return alpha_param,beta_param