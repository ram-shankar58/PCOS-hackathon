# test.py

import torch
from bindsnet.encoding import PoissonEncoder
from snnmodel import create_network
import numpy as np
import os
import cv2

# Set up parameters
time = 500  # time of simulation in ms
n_input = 280*280  # size of input layer
n_neurons = 100  # size of neuron layer
def load_data(X):
    p="Dataset/PCOSGen-train/PCOSGen-train/images/"
    return os.path.join(p,X)

vfunc = np.vectorize(load_data)

X_train = np.load('Dataset/X_train.npy',allow_pickle=True)
y_train = np.load('Dataset/y_train.npy', allow_pickle=True)
X_test = np.load('Dataset/X_test.npy', allow_pickle=True)
y_test = np.load('Dataset/y_test.npy', allow_pickle=True)

X_train =vfunc(X_train)
X_test = vfunc(X_test)

# Create the network
network, monitor = create_network(time, n_input, n_neurons)

# Create a PoissonEncoder to encode image data
encoder = PoissonEncoder(time=time)

# Testing
correct = 0
for image, label in zip(X_test, y_test):
    # Encode the image into spike trains
    image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    image = torch.from_numpy(image)

    input_data = encoder(image.view(-1))
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
