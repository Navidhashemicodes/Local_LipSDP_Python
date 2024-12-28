# MATLAB Implementation of Local-LipSDP toolbox

This toolbox  computes the local value of robustness certificates (Lipschitz constant) of deep neural networks. This toolbox is an extension of the [LipSDP](https://github.com/mahyarfazlyab/LipSDP) toolbox which provides the global robustness certificate (Lipschitz constant).

This research work is published [here](https://proceedings.mlr.press/v144/hashemi21a.html) and was presented as a part of L4DC 2021 conference.

## Instalation

There is no need to install the toolbox and you are only required to locate the repository somewhere in your machine. you may run:

**`git clone https://github.com/Navidhashemicodes/Local_LipSDP_Matlab.git`**

Once you located the toolbox in your machine and run an experiment yalmip will be added to your path automatically. However please run "yalmiptest" on your command window to make it certain yalmip works. You can also use cvx as an alternative.

We utilized mosek solver in our experiments, in case you have no access to the license you can easily replace it with another solver when you run the code.
   
## Requirements
MATLAB

YLAMIP or CVX (Preferably YALMIP)

MOSEK (optional)



## Example

To replicate the results of the [paper](https://proceedings.mlr.press/v144/hashemi21a/hashemi21a.pdf). The folder **`examples`** is provided. 

## Citation

@inproceedings{hashemi2021certifying,
  
title={Certifying incremental quadratic constraints for neural networks via convex optimization},
  
author={Hashemi, Navid and Ruths, Justin and Fazlyab, Mahyar},
  
booktitle={Learning for Dynamics and Control},
  
pages={842--853},
  
year={2021},
  
organization={PMLR}

}



