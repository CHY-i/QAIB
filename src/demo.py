import torch
import torch.nn as nn
from .data import generate_sureface_data
from .utils import surface_data_to_3D
from .model import MLP
from .seq_model import RecurrentNeuralNetwork, GatedRecurrentUnit, LongShortTermMemory, RecurrentNeuralNetworkDecoders

def demo():
    # Generate surface code data
    d = 3
    rounds = 5
    error_rate = 0.1
    num_samples = 10

    data = generate_sureface_data(d, rounds, error_rate, num_samples)
    print("Generated surface code data shape:", data.shape)

    # Convert surface data to 3D format
    data_3D = surface_data_to_3D(data)
    print("Converted 3D data shape:", data_3D.shape)

    # Define a simple MLP model
    n_in = (d + 1) * (d + 1) * (rounds + 1)
    n_out = 2
    n_hiddens = 64
    depth = 3

    mlp_model = MLP(n_in, n_out, n_hiddens, depth, activator='relu')
    print("MLP model structure:", mlp_model)

    # Forward pass through the MLP model
    sample_input = data_3D.view(num_samples, -1)  # Flatten the input
    output = mlp_model(sample_input)
    print("MLP model output shape:", output.shape)

    # Define a simple RNN model
    input_size = n_in
    hidden_size = 128
    rnn_network = nn.Linear(input_size + hidden_size, hidden_size)
    rnn_model = RecurrentNeuralNetwork(rnn_network)
    print("RNN model structure:", rnn_model)

    # Forward pass through the RNN model
    h_0 = torch.zeros(num_samples, hidden_size)
    rnn_output = rnn_model(sample_input, h_0)
    print("RNN model output shape:", rnn_output.shape)