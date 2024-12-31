Run the following file:

**`main-bisection.py`**

This script processes the data point stored in **`Input_data.pt`** and the model **`Trained_model.pt`**, which has been trained on the MNIST dataset. For a given uncertainty represented by an L-infinity ball with radius **ε**, the script determines the maximum allowable value of **ε** such that the model **`Trained_model.mat`** can still reliably classify the image **`Input_data.pt`** accurately, even when the image is perturbed by a noise within the L-infinity ball.


