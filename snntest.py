# test.py

from bindsnet import torch
from bindsnet.encoding import PoissonEncoder
from snnmodel import create_network


# Set up parameters
time = 500  # time of simulation in ms
n_input = 28*28  # size of input layer
n_neurons = 100  # size of neuron layer

import numpy as np

X_train = np.load('Dataset/X_train.npy')
y_train = np.load('Dataset/y_train.npy')
X_test = np.load('Dataset/X_test.npy')
y_test = np.load('Dataset/y_test.npy')

# Create the network
network, monitor = create_network(time, n_input, n_neurons)

# Create a PoissonEncoder to encode image data
encoder = PoissonEncoder(time=time)

# Testing
correct = 0
for image, label in zip(X_test, y_test):
    # Encode the image into spike trains
    input_data = encoder(image.view(28*28))
    inputs = {'Input Layer': input_data}

    # Run the network on the spike-encoded inputs
    network.run(inputs=inputs, time=time)

    # Get the output spikes
    output_spikes = monitor.get('s')

    # Check if the prediction is correct
    correct += int(torch.sum(output_spikes).item() > 0) == label

# Calculate the accuracy
accuracy = correct / len(X_test)
print('Accuracy:', accuracy)
