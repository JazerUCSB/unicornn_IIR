import torch
import torch.nn as nn
import torch.nn.functional as F

class UnICORNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dt, alpha, layers):
        super(UnICORNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.dt = dt
        self.alpha = alpha
        
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.recurrent_layers = nn.ModuleList([
            UnICORNN_Layer(hidden_dim, dt, alpha) for _ in range(layers)
        ])
        self.output_layer = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.input_layer(x)
        for layer in self.recurrent_layers:
            x = layer(x)
        x = self.output_layer(x)
        return x


class UnICORNN_Layer(nn.Module):
    def __init__(self, hidden_dim, dt, alpha):
        super(UnICORNN_Layer, self).__init__()
        self.hidden_dim = hidden_dim
        self.dt = dt
        self.alpha = alpha
        self.W = nn.Parameter(torch.randn(hidden_dim))
        self.V = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x):
        if x.shape[1] != self.hidden_dim:
            x = torch.cat([x] * (self.hidden_dim // x.shape[1]), dim=1)  # Expand input to match hidden_dim

        z = torch.zeros_like(x)
        for _ in range(x.shape[1]):  # Iterate over time steps
            z = z - self.dt * (torch.tanh(self.W * x + self.V @ x.T + self.b) + self.alpha * x)
            x = x + self.dt * z
        return x
