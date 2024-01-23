from typing import List, Tuple
# model.py

import torch
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection
from bindsnet.network.monitors import Monitor
from bindsnet.learning import PostPre

def create_network(time: int, n_input: int, n_neurons: int, update_rule=None, attributes_to_monitor=['v'], input_layer_name='Input Layer', neuron_layer_name='Neuron Layer', monitor_name='Monitor') -> Tuple[Network, Monitor]:
    '''
    Creates a neural network using the BindsNET library.

    Args:
        time: The simulation time in timesteps.
        n_input: The number of neurons in the input layer.
        n_neurons: The number of LIF neurons in the neuron layer.
        update_rule: The update rule for the connection.
        attributes_to_monitor: The attributes to monitor in the monitor.
        input_layer_name: The name of the input layer.
        neuron_layer_name: The name of the neuron layer.
        monitor_name: The name of the monitor.

    Returns:
        A neural network object.'''
    
    # Check if n_input and n_neurons are positive integers
    if not isinstance(n_input, int) or n_input <= 0:
        raise ValueError("n_input must be a positive integer.")
    if not isinstance(n_neurons, int) or n_neurons <= 0:
        raise ValueError("n_neurons must be a positive integer.")

    # Check if time is a positive integer
    if not isinstance(time, int) or time <= 0:
        raise ValueError("time must be a positive integer")

    # Check if input_layer_name, neuron_layer_name, and monitor_name are non-empty strings
    if not isinstance(input_layer_name, str) or input_layer_name == '':
        raise ValueError("input_layer_name must be a non-empty string")
    if not isinstance(neuron_layer_name, str) or neuron_layer_name == '':
        raise ValueError("neuron_layer_name must be a non-empty string")
    if not isinstance(monitor_name, str) or monitor_name == '':
        raise ValueError("monitor_name must be a non-empty string")

    # Set up the network
    network = Network()

    # Create layers of neurons
    input_layer = Input(n=n_input)
    neuron_layer = LIFNodes(n=n_neurons)

    # Connect the layers with customizable update rule
    if update_rule is None:
        update_rule = PostPre
    connection = Connection(
        source=input_layer, target=neuron_layer, w=torch.rand(n_input, n_neurons), update_rule=update_rule
    )

    # Add layers to the network with customizable names
    network.add_layer(input_layer, name=input_layer_name)
    network.add_layer(neuron_layer, name=neuron_layer_name)

    # Add connection to the network
    network.add_connection(connection, source=input_layer_name, target=neuron_layer_name)

    # Validate the attributes_to_monitor list
    valid_attributes = ['v', 's', 'a']
    for attr in attributes_to_monitor:
        if attr not in valid_attributes:
            raise ValueError(f"Invalid attribute: {attr}. Valid attributes are: {valid_attributes}")

    # Create a monitor with validated attributes to monitor
    monitor = Monitor(neuron_layer, attributes_to_monitor, time=time)
    network.add_monitor(monitor, name=monitor_name)

    return network,monitor
