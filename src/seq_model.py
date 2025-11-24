import torch
import torch.nn as nn
from .module import PositionalEmbedding

class RecurrentNeuralNetwork(nn.module):
    def __init__(self, network):
        super().__init__()
        self.network = network # (input_size to hidden_size)

    def forward(self, x, h):
        # [batch_size, input_size + hidden_size]
        combined = torch.cat((x, h), dim=1)
        # [batch_size, hidden_size]
        y = self.network(combined)
        h_next = torch.tanh(y)
        return h_next

class GatedRecurrentUnit(nn.Module):
    def __init__(self, network):
        super().__init__()
        self.network = network # (input_size to 3 (num_gates) * hidden_size)

    def forward(self, x, h):
        # [batch_size, input_size + hidden_size]
        combined = torch.cat((x, h), dim=1)
        gate_outputs = self.network(combined)
        
        #(r, z, n)
        r_gate, z_gate, n_gate = torch.chunk(gate_outputs, 3, dim=1)
        
        r = torch.sigmoid(r_gate) 
        z = torch.sigmoid(z_gate) 
        n = torch.tanh(n_gate + r * h)    
        
        # h_new = (1 - z) * n + z * h
        h_next = (1 - z) * n + z * h
        
        return h_next

class LongShortTermMemory(nn.Module):
    def __init__(self, network):
        super().__init__()
        self.network = network # (input_size to 4 (num_gates) * hidden_size)

    def forward(self, x, h_state):
        # h_state : (h, c)
        h, c = h_state
        
        combined = torch.cat((x, h), dim=1)
        
        gate_outputs = self.network(combined)
        
        f_gate, i_gate, o_gate, g_gate = torch.chunk(gate_outputs, 4, dim=1)
        
        f = torch.sigmoid(f_gate)
        i = torch.sigmoid(i_gate)
        o = torch.sigmoid(o_gate) 
        g = torch.tanh(g_gate)  
        
        # c_new = f * c_old + i * g
        c_next = f * c + i * g
        
        # h_new = o * tanh(c_new)
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next

class RecurrentNeuralNetworkDecoder(nn.Module):
    def __init__(self, rnn_type, network, hidden_size, output_size):
        """
        Args:
            rnn_type: 'rnn', 'gru', 'lstm'
            network: nn.Module, the underlying network for RNN cells
            hidden_size: int
            output_size: int
        """
        super().__init__()
        if rnn_type == 'rnn':
            rnn_block = RecurrentNeuralNetwork(network)
        elif rnn_type == 'gru':
            rnn_block = GatedRecurrentUnit(network)
        elif rnn_type == 'lstm':
            rnn_block = LongShortTermMemory(network)
        else:
            raise ValueError(f"Unsupported rnn_type: {rnn_type}")
        self.rnn_block = rnn_block
        self.classifier = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size

    def forward(self, x):
        # x shape: [batch_size, seq_length, feature_dim]
        batch_size, seq_length, _ = x.shape
        h = torch.zeros(batch_size, self.hidden_size).to(x.device)
        c = torch.zeros(batch_size, self.hidden_size).to(x.device) # only for LSTM
        outputs = []

        # Time Step Loop
        for t in range(seq_length):
            x_t = x[:, t, :]
            if isinstance(self.rnn_block, LongShortTermMemory):
                h, c = self.rnn_block(x_t, (h, c))
            else:
                h = self.rnn_block(x_t, h)
            
            outputs.append(h)

        # list of [batch, hidden] -> [batch, seq_length, hidden]
        outputs = torch.stack(outputs, dim=1)
        
        # [batch, seq_length, hidden] -> [batch, seq_length, output_size]
        x = self.classifier(outputs)
        return x