import torch
from torch import nn
import torch.optim as optim
from torch_geometric.data import Data
from torch.utils.data import Dataset, DataLoader, Sampler
import torch.nn.functional as F
import numpy as np
from encoder_batch import GCNEDModel, GCNEncoder, GCNDecoder
from ldm_predic import DiffusionModelWithPredicDecoder
from train1 import loaddata, process_d, process_edges, normalize, denormalize
from gcnlstm import *
import matplotlib.pyplot as plt

#pip install torch pandas torch_geometric matplotlib
device = torch.device("cpu") #"cuda" if torch.cuda.is_available() else "cpu"

batch_size = 1
num_hours = 12
num_nodes = 100
num_features = 1

# Random input tensor shaped like your real data
x = torch.randn(batch_size, num_hours, num_nodes, num_features).to(device)

# Fake adjacency matrix (you should replace with actual one from edge_sample)
adj = torch.eye(num_nodes).to(device)  # simple identity matrix as adjacency
edge_index = adj.nonzero(as_tuple=False).t().contiguous()

# Init encoder & decoder
encoder = GCNEncoder(in_channels=1, hidden_channels=16).to(device)
decoder = GCNDecoder(input_dim=1, hidden_dim=16, output_dim=1).to(device);

encoder.eval()
decoder.eval()

# Run through encoder and decoder
with torch.no_grad():
    print("Encoding...")
    latent = encoder(x[:, 0, :, :], edge_index)  # Just use 1st hour for testing
    print("Latent shape:", latent.shape)

    print("Decoding...")
    reconstructed = decoder(latent, edge_index)
    print("Reconstructed shape:", reconstructed.shape)

# Optional: Compare input vs output
print("Input (first 5 values):", x[0, 0, :5, 0])
print("Reconstructed (first 5 values):", reconstructed[0, :5, 0])

mse = F.mse_loss(reconstructed, x[:, 0, :, :])
print("MSE between input and reconstructed:", mse.item())

ha = input("Type to unfreeze")
# Load data
speed = torch.tensor(loaddata("/Users/liamseidel/Documents/STGAIL-main/speed_SZ.csv")).to(device)
demand = torch.tensor(loaddata("/Users/liamseidel/Documents/STGAIL-main/demand_SZ.csv")).to(device)
inflow = torch.tensor(loaddata("/Users/liamseidel/Documents/STGAIL-main/demand_SZ.csv")).to(device)
edge = np.load('adjacency_matrices_30.npy')

# Process data
speed, min_s, max_s, demand, min_d, max_d, inflow, min_i, max_i = process_d(speed, demand, inflow)
x = inflow.reshape(-1, 12, 63, 100, 1)
x = x[:, :, :50, :, :]

edge = torch.tensor(edge).float()  # Transform data
edge = process_edges(edge)
edge_expanded = edge[np.newaxis, np.newaxis, :, :, :]
edge_tiled = np.tile(edge_expanded, (162, 1, 1, 1))
edge_tiled = torch.tensor(edge_tiled).to(device)
edge_tiled = edge_tiled.reshape(162, 63, 100, 100)
edge_tiled = edge_tiled[:, :50, :, :]

# Choose one sample (e.g., region 0 from first sample)
region_id = 0
sample_id = 0
x_sample = x[sample_id:sample_id+1, :, region_id, :, :]  # shape: (1, 12, 100, 1)
print(x_sample.size())
edge_sample = edge_tiled[sample_id, region_id, :, :]  # shape: (100, 100)
print(edge_sample.size())
edge_index = edge_sample.nonzero(as_tuple=False).t().contiguous()

# Generate mask for prediction (e.g., mask the last 2 timesteps)
mask = torch.ones(1, 12).to(device)
mask[:, -2:] = 0  # mask last 2 timesteps

# Load models
model = GCNEDModel(1, 16, 1).to(device)

model.encoder.load_state_dict(torch.load("posttrain/model_encoder_checkpoint_epoch80.pth", map_location=torch.device("cpu")) )
model.decoder.load_state_dict(torch.load("posttrain/model_decoder_checkpoint_epoch80.pth",  map_location=torch.device("cpu")) )
model.eval()

diff_model = DiffusionModelWithPredicDecoder(
    in_channels=1, num_timesteps=1000, device=device, num_features=1, hidden_dim=64).to(device)
diff_model.load_state_dict(torch.load("/Users/liamseidel/Documents/STGAIL-main/549STGAIL/model/posttrain/model_checkpoint_mask2_ts1000_epoch25.pth", map_location=torch.device("cpu")) )
diff_model.eval()

# Run prediction
masked_x = x_sample * mask.unsqueeze(-1).unsqueeze(-1) #remove mask for step 2 (just x_sample)
latent = model.encoder(x_sample, edge_index) # masked_x -> x_sample
pred = diff_model(latent, edge_sample, mask) #skip this for test 2 stage for encoder/decoder
decoded = model.decoder(pred, edge_index)

# Visualize prediction vs. ground truth
ground_truth = x_sample.squeeze(-1).detach().cpu().numpy() #ground_truth = masked_x.squeeze(-1).cpu().detach()
prediction = decoded.squeeze(-1).detach().cpu().numpy() #prediction = decoded.squeeze(-1).cpu().detach()

#gt = (denormalize(ground_truth, min_i, max_i)).numpy()
#pred = (denormalize(prediction, min_i, max_i)).numpy()


loss = np.mean(np.square(np.subtract(ground_truth, prediction)))
print(loss)
#for hour in range(12):

    #plt.figure()
    #plt.title(f"Hour {hour}")
    #plt.imshow(ground_truth[0, hour].reshape(10,10), cmap="viridis", label="GT")
    #plt.colorbar()
    #plt.show()
    #plt.figure()
    #plt.title(f"Hour {hour}")
    #plt.imshow(prediction[0, hour].reshape(10,10), cmap="viridis", label="Predicted")
    #plt.colorbar()
    #plt.show()
