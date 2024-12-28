# Experiments

## Experiment 1
In this experiment, we consider an L-infinity ball centered at the origin with radius ε. We work with random neural networks of the same architecture and calculate the ratio between the local and global Lipschitz constants when the L-infinity ball defines the set of inputs. The experiment assumes an ascending order for ε, and we plot the ratio as a function of ε. The folder **`Experiment1`** contains the necessary files for this experiment.

## Experiment 2
This experiment uses the MNIST dataset, where we train a model to classify the images. For a given image sampled from the dataset, we compute the uncertainty level in terms of an L-infinity ball with radius ε that still ensures correct classification. This uncertainty level is represented by the radius ε, which we calculate using Local-LipSDP. Additionally, we demonstrate that when using LipSDP, the computed ε is smaller, meaning Local-LipSDP provides a less conservative robustness guarantee for the model's performance under uncertainty. The folder **`MNIST_robustness`** contains the relevant files for this experiment.

## Experiment 3
In this experiment, we compare the conservatism of the robustness certificates obtained through our method with other approaches such as LipOPT, LipMIP, CLEVER, FastLip, and LipLP. We use a small radius ε for the L-infinity ball centered at the origin and evaluate the Lipschitz constants computed by each method as ε increases. This experiment shows that Local-LipSDP exhibits significantly lower conservatism compared to FastLip, LipOPT, and LipLP. Although LipMIP also offers lower conservatism, it is not scalable. The folder **`comparing`** contains the files for this experiment.


The m-file **`random_test.m`** is provided as toy example to check the performance of the toolbox.
 


