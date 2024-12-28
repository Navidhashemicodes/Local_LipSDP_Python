# Experiment 3

## Figure 5-a

This figure presents a comparison between Local-LipSDP and LipOPT on a network with the structure **[20, 20, 202, 1]**. The folder **`Figure_5a`** contains the files used for this comparison. This folder is the open-source LipOPT repository which we also added the following files:

1. Comparison.mat  
2. Comparison.pt  
3. Local_LipSDP_result.m  
4. Untitled.ipynb  

The files Comparison.pt and **`Comparison.mat`** represent a unique neural network in PyTorch and MAT-file formats, respectively. The former is used in **`Untitled.ipynb`** to compute the Local Lipschitz constants, while the latter is used in **`Local_LipSDP_result.m`** to compute the local Lipschitz constants, which are then compared with LipOPT. The Lipschitz constants are generated across a unique range of ε, which represents the radius of the L-infinity ball. By the way, the computation runtime for LipOPT is at least 20 times higher than Local-LipSDP. Please run the mfile and ipynb file to check the computation runtime. This is while Local-LipSDP is provides significantly a lower level of conservatism.

## Figure 5-b

This figure presents a comparison between Local-LipSDP and several other methods, including LipMIP, LipLP, RandomLB, CLEVER, and FastLip, on a network with the structure **[2, 100, 100, 2]**. The CLEVER and RandomLB techniques do not provide guarantees and are used solely as benchmarks. Additionally, LipOPT was excluded from the comparison due to scalability issues in generating its Gurobi-optimization file for a model of this structure. 

The folder **`Figure_5b`** contains the files used for this comparison. This folder is based on the open-source LipMIP repository, which already includes source codes for comparing LipLP, RandomLB, CLEVER, and FastLip. The following additional files are also included:

1. 2by2comparison.mat  
2. 2by2comparison.pt  
3. Local_LipSDP_result.m  
4. tutorial3.ipynb  

The files **`2by2comparison.pt`** and **`2by2comparison.mat`** represent a unique neural network in PyTorch and MAT-file formats, respectively. All these files are located in the directory: **`Figure_5b\lipMIP-master\tutorials`**.

In this experiment, we assume a specific range of radius values, **ε**, for the L-infinity ball and compute the local Lipschitz constants for each radius ε using all the provided techniques. The file **`tutorial3.ipynb`** uses **`2by2comparison.pt`** to perform the computations for LipMIP, LipLP, RandomLB, CLEVER, and FastLip, while **`Local_LipSDP_result.m`** uses **`2by2comparison.mat`** to perform the computations for Local-LipSDP.
  



