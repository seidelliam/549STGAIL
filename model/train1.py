import os
import random
import time
import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch_geometric.data import Data
from torch.utils.data import Dataset, DataLoader
from ldm_predic import *
from encoder_batch import *
from ddpm import DiffusionModel
import matplotlib.pyplot as plt


# run second use to run the encoder decoder, generating predictions, modifying prediction, write own testing code and data?

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
    return normalized_data, min_val, max_val


def process_d(speed, demand, inflow):
    demand_threshold = torch.quantile(demand, 0.9)
    inflow_threshold = torch.quantile(inflow, 0.9)

    speed_np = torch.clamp(speed, max=140)
    demand_np = torch.clamp(demand, max=demand_threshold)
    inflow_np = torch.clamp(inflow, max=inflow_threshold)

    x1, dem_min, dem_max = normalize(demand_np)
    y1, inf_min, inf_max = normalize(inflow_np)
    z1, spe_min, spe_max = normalize(speed_np)

    temp2 = y1.unsqueeze(1)
    temp1 = x1.unsqueeze(1)
    temp3 = z1.unsqueeze(1)

    res_speed = temp3.reshape(-1, 63, 100).float()
    res_demand = temp1.reshape(-1, 63, 100).float()
    res_inflow = temp2.reshape(-1, 63, 100).float()

    return res_speed, spe_min, spe_max, res_demand, dem_min, dem_max, res_inflow, inf_min, inf_max


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


def load_pretrained_weights(diff_model, pretrained_path):
    pretrained_dict = torch.load(pretrained_path, map_location=torch.device('cpu'))
    model_dict = diff_model.state_dict()

    # Filter keys to match the current model
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    diff_model.load_state_dict(model_dict)


def generate_mask(batch_size, num_hours, mask_num):
    mask = torch.ones(batch_size, num_hours)
    zero_indices = torch.randperm(num_hours)[:mask_num]
    mask[:, zero_indices] = 0
    return mask


def denormalize(normalized_data, min_val, max_val):
    return (normalized_data + 1) * (max_val - min_val) / 2 + min_val


def save_model(epoch, model, mask_num, time_step, path="saved_checkpoints/model_checkpoint_mask{}_ts{}_epoch{}.pth"):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path.format(mask_num, time_step, epoch))


def main():
    device = torch.device("cuda")
    tracked_loss = []
    # torch.cuda.set_device(1)  # Use GPU 1
    # print(f"Using device: {torch.cuda.get_device_name(device)}")

    speed = loaddata('speed_SZ.csv')
    demand = loaddata('demand_SZ.csv')
    inflow = loaddata('inflow_SZ.csv')
    edge = np.load('adjacency_matrices_30.npy')  # For speed

    speed = torch.tensor(speed).to(device)  # Convert numpy array to tensors
    inflow = torch.tensor(inflow).to(device)
    demand = torch.tensor(demand).to(device)

    speed, s_min, s_max, demand, d_min, d_max, inflow, i_min, i_max = process_d(speed, demand, inflow)
    inflow = inflow.reshape(-1, 12, 63, 100, 1)  # x = inflow.reshape(-1, 12, 63, 100, 1)
    speed = speed.reshape(-1, 12, 63, 100, 1)  # reshaping speed
    speed = speed[:, :, :50, :, :]  # x = speed[:, :, :50, :, :]

    edge = torch.tensor(edge).float()  # Transform data
    edge = process_edges(edge)
    edge_expanded = edge[np.newaxis, np.newaxis, :, :, :]
    edge_tiled = np.tile(edge_expanded, (162, 1, 1, 1))
    edge_tiled = torch.tensor(edge_tiled).to(device)
    edge_tiled = edge_tiled.reshape(162, 63, 100, 100)
    edge_tiled = edge_tiled[:, :50, :, :]

    batch = 32
    mask_num = 2

    dataset = MyDataset(speed, edge_tiled)
    dataloader = DataLoader(dataset, batch_size=batch, shuffle=True)

    in_channels = 1
    hidden_channels = 16
    out_channels = 1

    num_timesteps = 1000
    num_features = 1

    model = GCNEDModel(in_channels, hidden_channels, out_channels).to(device)

    model.encoder.load_state_dict(torch.load(
        "saved_checkpoints/model_encoder_checkpoint_epoch80.pth",
        map_location=torch.device('cpu')
    ), strict=False)
    model.decoder.load_state_dict(torch.load(
        "saved_checkpoints/model_decoder_checkpoint_epoch80.pth",
        map_location=torch.device('cpu')
    ), strict=False)

    diff_model = DiffusionModelWithPredicDecoder(
        in_channels=1, num_timesteps=1000, device=device, num_features=1, hidden_dim=64).to(device)
    load_pretrained_weights(diff_model, "saved_checkpoints/model_checkpoint_ts200_epoch30.pth")

    optimizer = optim.Adam(diff_model.parameters(), lr=0.00001)
    criterion_recon = nn.MSELoss()

    for epoch in range(25):  # originally 1000
        model.train()
        diff_model.train()
        total_loss = 0

        for batch_idx, (data, edge) in enumerate(dataloader):
            print("Processing batch " + str(batch_idx))

            batch_loss = 0
            data = data.to(device)
            edge = edge.to(device)

            optimizer.zero_grad()

            batch_size, num_hours, num_regions, num_nodes, num_feature = data.shape

            region_list = np.arange(num_regions)
            np.random.shuffle(region_list)

            for i in region_list:
                data_region = data[:, :, i, :, :]
                data_edge = edge[0, i, :, :]

                mask = generate_mask(batch_size, num_hours, mask_num).to(device)  # mask what we are trying to predict
                data_regionx = data_region * mask.unsqueeze(-1).unsqueeze(-1)
                edge_indices = data_edge.nonzero(as_tuple=False).t().contiguous().to(device)

                embed_data = model.encoder(data_regionx, edge_indices)
                recover_data = diff_model(embed_data, data_edge, mask)
                prediction = model.decoder(recover_data, edge_indices)
                masked_data_region = data_region * (1 - mask.unsqueeze(-1).unsqueeze(-1))
                masked_prediction = prediction * (1 - mask.unsqueeze(-1).unsqueeze(-1))
                loss = criterion_recon(masked_prediction, masked_data_region)
                loss.backward(retain_graph=True)

                batch_loss += loss.item()

            optimizer.step()

            total_loss += batch_loss

        loss = total_loss / (len(dataloader) * num_regions)
        if (epoch + 1) % 25 == 0:
            save_model(epoch + 1, diff_model, mask_num, num_timesteps)
        print(f'Epoch {epoch + 1}, Loss: {loss:.4f}')
        tracked_loss.append(loss)


if __name__ == '__main__':
    main()
