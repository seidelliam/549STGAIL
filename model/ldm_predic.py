import torch
from torch import nn
import torch.nn.functional as F
import sys
from torch_geometric.nn import GCNConv
from model import *
from model import Model
from util import make_beta_schedule, extract_into_tensor

sys.path.append('../')

#from LDM.util import *
#from LDM.util import *

class FineTuneModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FineTuneModel, self).__init__()
        self.conv1 = GCNConv(input_dim + 1, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, mask, edge_index):
        batch_size, num_hours, num_nodes, num_features = x.shape
        outputs = torch.zeros(batch_size, num_hours, num_nodes, num_features, device=x.device)

        for hour in range(num_hours):
            mask_channel = mask[:, hour].unsqueeze(-1).unsqueeze(-1).expand(-1, num_nodes, 1)
            masked_input = torch.cat((x[:, hour, :, :], mask_channel), dim=-1)

            encoded = self.conv1(masked_input, edge_index)
            decoded = self.conv2(encoded, edge_index)
            output = torch.tanh(decoded)

            outputs[:, hour, :, :] = output

        return outputs

class DiffusionModelWithPredicDecoder(nn.Module):
    def __init__(self, in_channels, num_timesteps, device, num_nodes=100, num_features=1, hidden_dim=64):
        super().__init__()

        self.model = Model(in_channels, num_features, edge_importance_weighting=False)
        self.predic_decoder = FineTuneModel(in_channels, hidden_dim, num_features)
        self.num_timesteps = num_timesteps
        self.schedule = "cosine"
        self.device = device

        self.betasx = make_beta_schedule(self.schedule, self.num_timesteps)
        if not hasattr(self, 'betasx'):
            self.register_buffer('betasx', self.betasx)

        self.alphas = 1 - self.betasx
        if not hasattr(self, 'alphas'):
            self.register_buffer('alphas', self.alphas)

        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / torch.cumprod(self.alphas, dim=0))
        if not hasattr(self, 'sqrt_recip_alphas_cumprod'):
            self.register_buffer('sqrt_recip_alphas_cumprod', self.sqrt_recip_alphas_cumprod)

        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - torch.cumprod(self.alphas, dim=0))
        if not hasattr(self, 'sqrt_one_minus_alphas_cumprod'):
            self.register_buffer('sqrt_one_minus_alphas_cumprod', self.sqrt_one_minus_alphas_cumprod)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start).to(x_start.device)

        t_tensor = torch.full((x_start.size(0),), t, dtype=torch.long, device=x_start.device)

        self.sqrt_recip_alphas_cumprod = self.sqrt_recip_alphas_cumprod.to(x_start.device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(x_start.device)

        noisy_data = (extract_into_tensor(self.sqrt_recip_alphas_cumprod, t_tensor, x_start.shape) * x_start +
                      extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t_tensor, x_start.shape) * noise)

        return noisy_data

    def forward(self, x, adjacency, mask):
        if x.dim() == 4:
            x = x.unsqueeze(1)

        t = torch.randint(0, self.num_timesteps, (1,), device=self.device).item()
        x_noisy = self.q_sample(x, t)

        x_recon = self.model(x_noisy, adjacency)

        edge_indices = adjacency.nonzero(as_tuple=False).t().contiguous().to(self.device)

        prediction = self.predic_decoder(x_recon, mask, edge_indices)

        return prediction

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
