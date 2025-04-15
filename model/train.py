import sys

# run first more of the training process, familiarize with history dist

# 3/27 Train:
# Start loss: 0.1040
# End Loss:
sys.path.append('../')
sys.path.append('../Encoder/')
sys.path.append('../Stgcn/')
import os
import matplotlib.pyplot as plt
import random
import time
import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch_geometric.data import Data
from torch.utils.data import Dataset, DataLoader, Sampler
from encoder_batch import *
from ddpm import *


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


def process_edges(edge):
    binary_edge = (edge != 0).float()
    return binary_edge


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tracked_loss = []
    # update the data path here
    speed = loaddata('speed_SZ.csv')
    demand = loaddata('demand_SZ.csv')
    inflow = loaddata('inflow_SZ.csv')
    edge = np.load('adjacency_matrices_30.npy')

    speed = torch.tensor(speed).to(device)
    inflow = torch.tensor(inflow).to(device)
    demand = torch.tensor(demand).to(device)

    speed, demand, inflow = process_d(speed, demand, inflow)
    speed = speed.reshape(-1, 12, 63, 100, 1)
    speed = speed[:, :, :50, :, :]
    edge = torch.tensor(edge).float()
    edge = process_edges(edge)
    edge_expanded = edge[np.newaxis, np.newaxis, :, :, :]
    edge_tiled = np.tile(edge_expanded, (162, 1, 1, 1))
    edge_tiled = torch.tensor(edge_tiled).to(device)
    edge_tiled = edge_tiled.reshape(162, 63, 100, 100)
    edge_tiled = edge_tiled[:, :50, :, :]

    batch = 32

    dataset = MyDataset(speed, edge_tiled)
    dataloader = DataLoader(dataset, batch_size=batch, shuffle=True)

    in_channels = 1
    hidden_channels = 16
    out_channels = 1

    num_timesteps = 200
    num_features = 1

    model = GCNEDModel(in_channels, hidden_channels, out_channels).to(device)
    model.train()
    model.encoder.load_state_dict(torch.load
                                  ("saved_checkpoints/model_encoder_checkpoint_epoch80.pth",
                                   map_location=torch.device('cuda')))  # weights_only -> True
    model.decoder.load_state_dict(torch.load
                                  ("saved_checkpoints/model_decoder_checkpoint_epoch80.pth",
                                   map_location=torch.device('cuda')))  # weights_only -> True

    diff = DiffusionModel(in_channels, num_timesteps, num_features=num_features, device=device).to(device)

    optimizer = optim.Adam(diff.parameters(), lr=0.00001)  # 0.0001 ->0.001
    criterion_recon = nn.MSELoss()

    print("reach the epoch")

    for epoch in range(20):  # originally 1000
        model.train()
        diff.train()
        total_loss = 0

        for batch_idx, (data, edge) in enumerate(dataloader):
            print("processing batch" + str(batch_idx))

            batch_loss = 0
            data = data.to(device)
            edge = edge.to(device)

            batch_size, num_hours, num_regions, num_nodes, num_feature = data.shape

            region_list = np.arange(num_regions)
            np.random.shuffle(region_list)

            for i in region_list:
                data_region = data[:, :, i, :, :]
                data_edge = edge[0, i, :, :]

                edge_indices = data_edge.nonzero(as_tuple=False).t().contiguous().to(device)
                optimizer.zero_grad()
                embed_data = model.encoder(data_region, edge_indices)
                recover_data = diff(embed_data, data_edge)
                de_data = model.decoder(recover_data, edge_indices)
                loss = criterion_recon(de_data, data_region)
                loss.backward(retain_graph=True)
                optimizer.step()
                batch_loss += loss.item()

            total_loss += batch_loss

        loss = total_loss / (len(dataloader) * num_regions)
        if (epoch + 1) % 20 == 0:  # just going to save after the number of epochs in the loop
            save_model(epoch + 1, diff, num_timesteps)
        print(f'Epoch {epoch + 1}, Loss: {loss:.4f}')
        tracked_loss.append(loss)

    plt.figure(figsize=(10, 5))
    plt.plot(tracked_loss, label='Training Loss', color='blue', linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Epoch Loss Curve")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def save_model(epoch, model, time_step, path="saved_checkpoints/model_checkpoint_ts{}_epoch{}.pth"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path.format(time_step, epoch))


if __name__ == '__main__':
    main()
