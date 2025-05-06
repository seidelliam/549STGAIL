import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from encoder_batch import GCNEDModel
from ldm_predic import DiffusionModelWithPredicDecoder
from train1 import loaddata, process_d, process_edges, denormalize

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and preprocess data
speed = torch.tensor(loaddata('/Users/liamseidel/Documents/STGAIL-main/speed_SZ.csv')).to(device)
demand = torch.tensor(loaddata('/Users/liamseidel/Documents/STGAIL-main/demand_SZ.csv')).to(device)
inflow = torch.tensor(loaddata('/Users/liamseidel/Documents/STGAIL-main/inflow_SZ.csv')).to(device)
edge = np.load('adjacency_matrices_30.npy')  # Shape: (63, 100, 100)

# Preprocess and reshape input (keep full 100 nodes)
speed, s_min, s_max, demand, d_min, d_max, inflow, i_min, i_max = process_d(speed, demand, inflow)
x = inflow.reshape(-1, 12, 63, 100, 1)  # Shape: (N, 12, 63, 100, 1)
edge = torch.tensor(edge).float()
edge = process_edges(edge)

# Tile edge data for each sample
edge_expanded = edge[np.newaxis, np.newaxis, :, :, :]
edge_tiled = np.tile(edge_expanded, (162, 1, 1, 1))  # 162 samples
edge_tiled = torch.tensor(edge_tiled).to(device).reshape(162, 63, 100, 100)

# Select one sample and region
sample_id = 0
region_id = 0
x_sample = x[sample_id:sample_id + 1, :, region_id, :, :]  # (1, 12, 100, 1)
edge_sample = edge_tiled[sample_id, region_id, :, :]       # (100, 100)
edge_index = edge_sample.nonzero(as_tuple=False).t().contiguous().to(device)

# Mask last 2 timesteps for prediction
mask = torch.ones(1, 12).to(device)
mask[:, -2:] = 0
masked_x = x_sample * mask.unsqueeze(-1).unsqueeze(-1)  # Shape: (1, 12, 100, 1)

# Load pretrained models
model = GCNEDModel(1, 16, 1).to(device)
model.encoder.load_state_dict(torch.load("saved_checkpoints/model_encoder_checkpoint_epoch80.pth", map_location=device))
model.decoder.load_state_dict(torch.load("saved_checkpoints/model_decoder_checkpoint_epoch80.pth", map_location=device))
model.eval()

diff_model = DiffusionModelWithPredicDecoder(
    in_channels=1, num_timesteps=1000, device=device, num_features=1, hidden_dim=64).to(device)
diff_model.load_state_dict(torch.load("saved_checkpoints/model_checkpoint_mask2_ts1000_epoch25.pth", map_location=device))
diff_model.eval()

# Run inference
with torch.no_grad():
    latent = model.encoder(masked_x, edge_index)
    recovered = diff_model(latent, edge_sample, mask)
    prediction = model.decoder(recovered, edge_index)
# Evaluate only on last 13 nodes (87 to 99)
ground_truth = x_sample.squeeze(-1).cpu().numpy()[:,:, 87:]     # (12, 13)
print(ground_truth.shape)
predicted = prediction.squeeze(-1).cpu().numpy()[:,:, 87:]      # (12, 13)
print(predicted.shape)

mse = np.mean((ground_truth - predicted) ** 2)
correlation = np.corrcoef(ground_truth.flatten(), predicted.flatten())[0, 1]

print(f"MSE on last 13 nodes: {mse:.6f}")
print(f"Correlation: {correlation:.4f}")

# Visualization for last hour
hour = 11
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.title("Ground Truth (Hour 11)")
plt.imshow(ground_truth[0,hour].reshape(1, 13), cmap="viridis", aspect='auto')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.title("Prediction (Hour 11)")
plt.imshow(predicted[0,hour].reshape(1, 13), cmap="viridis", aspect='auto')
plt.colorbar()

plt.tight_layout()
plt.show()


