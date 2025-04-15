import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn

class GraphConvLSTMCell(nn.Module):
    def __init__(self, node_features, hidden_dim, device, bias=True):
        super(GraphConvLSTMCell, self).__init__()

        self.node_features = node_features
        self.hidden_dim = hidden_dim
        self.bias = bias
        self.device = device

        self.gconv = pyg_nn.GCNConv(node_features + hidden_dim, 4 * hidden_dim, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        self.gconv.reset_parameters()

    def forward(self, x, edge_index, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([x, h_cur], dim=2)
        combined_conv = self.gconv(combined, edge_index)
        cc_i, cc_f, cc_o, cc_g = combined_conv.chunk(4, dim=2)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, num_nodes):
        return (torch.zeros(batch_size, num_nodes, self.hidden_dim, device=self.device),
                torch.zeros(batch_size, num_nodes, self.hidden_dim, device=self.device))

class GraphConvLSTM(nn.Module):
    def __init__(self, node_features, hidden_dim, num_layers, device, bias=True):
        super(GraphConvLSTM, self).__init__()

        self.node_features = node_features
        self.hidden_dim = [hidden_dim] * num_layers if isinstance(hidden_dim, int) else hidden_dim
        self.num_layers = num_layers
        self.bias = bias

        cell_list = []
        for i in range(num_layers):
            cur_input_features = node_features if i == 0 else self.hidden_dim[i - 1]
            cell_list.append(GraphConvLSTMCell(cur_input_features, self.hidden_dim[i], device, bias=bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, x, edge_index, hidden_state=None):
        if hidden_state is None:
            hidden_state = [cell.init_hidden(x.size(0), x.size(2)) for cell in self.cell_list]

        last_state_list = []
        cur_input = x

        for layer_idx, cell in enumerate(self.cell_list):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(cur_input.size(1)):
                h, c = cell(cur_input[:, t, :, :], edge_index, (h, c))
                output_inner.append(h)
            cur_input = torch.stack(output_inner, dim=1)
            last_state_list.append((h, c))

        return last_state_list[-1]

