import torch
from torch import nn
import torch.nn.functional as F
from util import *
import sys
sys.path.append('../')
from model import *

class DiffusionModel(nn.Module):
    def __init__(self, in_channels, num_timesteps, device, num_nodes=100, num_features=1):
        super().__init__()

        self.model = Model(in_channels, num_features, edge_importance_weighting=False)
        self.num_timesteps = num_timesteps
        self.schedule = "cosine"
        self.device = device

        self.betasx = make_beta_schedule(self.schedule, self.num_timesteps) #might need to be a betasx
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

    def q_sample(self, x_start, t, noise=None): #Adding noise
        if noise is None:
            noise = torch.randn_like(x_start).to(x_start.device)
        t_tensor = torch.full((x_start.size(0),), t, dtype=torch.long, device=x_start.device)
        self.sqrt_recip_alphas_cumprod = self.sqrt_recip_alphas_cumprod.to(x_start.device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(x_start.device)
        noisy_data = (extract_into_tensor(self.sqrt_recip_alphas_cumprod, t_tensor, x_start.shape) * x_start +
                      extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t_tensor, x_start.shape) * noise)
        return noisy_data

    def forward(self, x, adjacency): #feed encoder output to this class
        if x.dim() == 4:
            x = x.unsqueeze(1)

        t = torch.randint(0, self.num_timesteps, (1,), device=self.device).item()
        x_noisy = self.q_sample(x, t)
        x_recon = self.model(x_noisy, adjacency) #reconstruction of noisy input
        return x_recon

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
