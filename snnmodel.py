# model.py

import torch
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection
from bindsnet.network.monitors import Monitor
from bindsnet.learning import PostPre

def create_network(time: int, n_input: int, n_neurons: int, neuron_type: str) -> Network:
    """
    Creates a neural network using the BindsNET library.

    Args:
        time: The simulation time in timesteps.
        n_input: The number of neurons in the input layer.
        n_neurons: The number of neurons in the neuron layer.
        neuron_type: The type of neurons in the neuron layer.

    Returns:
        The created neural network.
    """
    # Set up the network
    network = Network()

    # Create layers of neurons
    input_layer = Input(n=n_input)
    
    if neuron_type == 'LIF':
        neuron_layer = LIFNodes(n=n_neurons)
    elif neuron_type == 'IF':
        neuron_layer = IFNodes(n=n_neurons)
    else:
        raise ValueError(f"Invalid neuron type: {neuron_type}")

    # Connect the layers
    connection = Connection(
        source=input_layer, target=neuron_layer, w=torch.rand(n_input, n_neurons), update_rule=PostPre
    )

    # Add layers to the network
    network.add_layer(input_layer, name='Input Layer')
    network.add_layer(neuron_layer, name='Neuron Layer')

    # Add connection to the network
    network.add_connection(connection, source='Input Layer', target='Neuron Layer')

    # Create a monitor
    monitor = Monitor(neuron_layer, ['v'], time=time)
    network.add_monitor(monitor, name='Monitor')

    return network
