import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Model(nn.Module):
    def __init__(self, in_channels, num_features, edge_importance_weighting, **kwargs):
        super().__init__()

        spatial_kernel_size = 1
        temporal_kernel_size = 5
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * 100)
        self.st_gcn_networks = nn.ModuleList([ #Sequential application of st_gcn layers
            st_gcn(in_channels, 64, kernel_size, 1, residual=False, **kwargs),
            st_gcn(64, 128, kernel_size, 1, **kwargs) #the dif in in/out channels allow for network to progressively capture more complex features
        ])

        if edge_importance_weighting: #dynamically changing the weights of each adjacency matrix
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size())) for _ in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        self.fcn = nn.Sequential( #upsample and transform data back after being passed through GCNs
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, num_features, kernel_size=3, stride=1, padding=1)
        )

        self.T = 12
        self.V = 100
        self.num_features = num_features

        fc_input_dim = self.T * self.V * 4 #reshapes our flattened image to our final target dimension
        fc_output_dim = self.T * self.V * self.num_features
        self.fc = nn.Linear(fc_input_dim, fc_output_dim)
        
    def forward(self, x, A):
        A = A.to(x.device) #shaping input data to the format needed for data_bn
        N, C, T, V, M = x.size()
        if self.T is None or self.V is None:
            self.T = T
            self.V = V

        x = x.permute(0, 4, 3, 1, 2).contiguous()#shaping input data to the format needed for gcn
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x = gcn(x, A * importance)

        x = self.fcn(x)
        x = x.view(N, 1, -1)
        x = self.fc(x)
        x = x.view(N, T, V, -1)
        return x

class st_gcn(nn.Module): #Spatial-Temporal Graph Convolutional Network
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dropout=0, residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size[1])

        self.tcn = nn.Sequential( #BatchNorm, ReLU, Conv2d are all different temporal layers
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True)
        )

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)
                ),
                nn.BatchNorm2d(out_channels)
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        res = self.residual(x)
        x = self.gcn(x, A)
        x = self.tcn(x) + res
        return self.relu(x)

class ConvTemporalGraphical(nn.Module): # This class performs the core graph convolution operation on the spatial dimension using the adjacency matrix A.
    def __init__(self, in_channels, out_channels, kernel_size, t_kernel_size=1, t_stride=1, t_padding=0, t_dilation=1, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)

    def forward(self, x, A):
        x = self.conv(x)
        x = torch.einsum('nctv,vw->nctw', x, A)
        return x
