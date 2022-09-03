import torch
import torch.nn as nn
import torch.nn.functional as F


class FCQ(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dims, activation=F.relu):
        super(FCQ, self).__init__()

        self.activation = activation

        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()

        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
        self.hidden_layers = nn.Sequential(*self.hidden_layers)

        self.out_layer = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, state):
        action_values = self.out_layer(self.activation(self.hidden_layers(self.activation(self.input_layer(state)))))
        return action_values
