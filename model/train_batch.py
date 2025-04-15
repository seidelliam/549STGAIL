import os
import random
import time
import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch_geometric.data import Data
from encoder_batch import *
from torch.utils.data import Dataset, DataLoader, Sampler

def loaddata(data_name):
    path = data_name
    data = pd.read_csv(path, header=None)
    return data.values.reshape(-1, 63, 10, 10)

def normalize(data):
    min_val = torch.min(data)
    max_val = torch.max(data)

    if max_val - min_val == 0:
        return torch.zeros_like(data)

    normalized_data = 2 * (data - min_val) / (max_val - min_val) - 1

    return normalized_data

def process_d(speed, demand, inflow):
    demand_threshold = torch.quantile(demand, 0.9)
    inflow_threshold = torch.quantile(inflow, 0.9)

    speed_np = torch.clamp(speed, max=140)
    demand_np = torch.clamp(demand, max=demand_threshold)
    inflow_np = torch.clamp(inflow, max=inflow_threshold)

    x1 = normalize(demand_np)
    y1 = normalize(inflow_np)
    z1 = normalize(speed_np)

    temp2 = y1.unsqueeze(1)
    temp1 = x1.unsqueeze(1)
    temp3 = z1.unsqueeze(1)

    res_speed = temp3.reshape(-1, 63, 100).float()
    res_demand = temp1.reshape(-1, 63, 100).float()
    res_inflow = temp2.reshape(-1, 63, 100).float()

    return res_speed, res_demand, res_inflow

class MyDataset(Dataset):
    def __init__(self, data, edge):
        self.data = data
        self.edge = edge

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.edge[index]

def main():
    device = torch.device("cuda")

    # load the dataset here, update the path
    speed = loaddata('speed_SZ.csv')
    inflow = loaddata('demand_SZ.csv')
    demand = loaddata('inflow_SZ.csv')
    edge = np.load('adjacency_matrices_30.npy')

    speed = torch.tensor(speed).to(device)
    inflow = torch.tensor(inflow).to(device)
    demand = torch.tensor(demand).to(device)

    speed, demand, inflow = process_d(speed, demand, inflow)
    speed = demand.reshape(-1, 12, 63, 100, 1)
    speed = speed[:, :, :50, :, :]

    edge = torch.tensor(edge).float()
    edge_expanded = edge[np.newaxis, np.newaxis, :, :, :]
    edge_tiled = np.tile(edge_expanded, (162, 1, 1, 1))
    edge_tiled = torch.tensor(edge_tiled)
    edge_tiled = edge_tiled.reshape(162, 63, 100, 100)
    edge_tiled = edge_tiled[:, :50, :, :]

    batch = 32

    dataset = MyDataset(speed, edge_tiled)
    dataloader = DataLoader(dataset, batch_size=batch, shuffle=True)

    in_channels = 1
    hidden_channels = 16
    out_channels = 1

    K = 3
    num_layers = 3
    input_dim = 1
    hidden_dim = 8

    model = GCNEDModel(in_channels, hidden_channels, out_channels).to(device)
    discriminator = GCNDiscriminator(input_dim, hidden_dim, num_layers, device).to(device)

    optimizer_g = optim.Adam(list(model.encoder.parameters()) + list(model.decoder.parameters()), lr=0.0001)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.001)
    criterion_recon = nn.MSELoss()
    criterion_disc = nn.BCELoss()

    print("reach the epoch")

    for epoch in range(400):
        model.train()
        discriminator.train()
        total_loss = 0

        for batch_idx, (data, edge) in enumerate(dataloader):
            print("processing batch" + str(batch_idx))
            data = data.to(device)
            edge = edge.to(device)

            batch_size, num_hours, num_regions, num_nodes, num_feature = data.shape
            region_list = np.arange(num_regions)
            np.random.shuffle(region_list)

            for i in region_list:
                data_region = data[:, :, i, :, :]
                data_edge = edge[0, i, :, :]

                edge_indices = data_edge.nonzero(as_tuple=False).t().contiguous()
                optimizer_g.zero_grad()
                optimizer_d.zero_grad()

                generated_data = model(data_region, edge_indices)
                loss_g = criterion_recon(generated_data, data_region)
                loss_g.backward(retain_graph=True)

                discriminator_output_fake = discriminator(generated_data, edge_indices)
                discriminator_output_real = discriminator(data_region, edge_indices)

                loss_d_real = criterion_disc(discriminator_output_real, torch.ones_like(discriminator_output_real))
                loss_d_fake = criterion_disc(discriminator_output_fake, torch.zeros_like(discriminator_output_fake))
                loss_d = loss_d_real + loss_d_fake
                loss_d.backward()

                optimizer_d.step()
                optimizer_g.step()
                
                total_loss += loss_g.item() + loss_d.item()

        loss = total_loss / (len(dataloader) * num_regions)
        if (epoch + 1) % 100 == 0:
            save_model(epoch + 1, model.encoder, path="saved_checkpoints/model_encoder_checkpoint.pth")
            save_model(epoch + 1, model.decoder, path="saved_checkpoints/model_decoder_checkpoint.pth")
        print(f'Epoch {epoch + 1}, Loss: {loss:.4f}')

def save_model(epoch, model, path="model_checkpoint.pth"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path.format(epoch))

if __name__ == '__main__':
    main()

