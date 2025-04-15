import torch
import numpy as np
from torch_geometric.nn import GCNConv
from torch import nn
import torch.nn.functional as F

from gcnlstm import *

class GCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, in_channels)

    def forward(self, x, edge_index):
        x1 = self.conv1(x, edge_index)
        x2 = self.conv2(x1, edge_index)
        x3 = torch.tanh(x2)
        return x3

class GCNDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCNDecoder, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x1 = self.conv1(x, edge_index)
        x2 = self.conv2(x1, edge_index)
        x3 = torch.tanh(x2)
        return x3

class GCNDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, device, bias=True):
        super(GCNDiscriminator, self).__init__()
        self.device = device
        self.gconv_lstm = GraphConvLSTM(input_dim, hidden_dim, num_layers, device, bias=bias)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index):
        final_hidden_state = self.gconv_lstm(x, edge_index)
        final_score = self.fc(final_hidden_state[0])
        final_score = torch.sigmoid(final_score)
        return final_score

class GCNEDModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNEDModel, self).__init__()
        self.encoder = GCNEncoder(in_channels, hidden_channels)
        self.decoder = GCNDecoder(in_channels, hidden_channels, out_channels)

    def forward(self, x, edge_index):
        batch, num_hours, num_nodes, num_feature = x.shape
        outputs = torch.zeros(batch, num_hours, num_nodes, num_feature, device=x.device)

        for hour in range(num_hours):
            region_data = x[:, hour, :]
            encoded = self.encoder(region_data, edge_index)
            decoded = self.decoder(encoded, edge_index)
            outputs[:, hour] = decoded

        return outputs

