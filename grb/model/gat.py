import torch
import torch.nn as nn
import torch.nn.functional as F


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""

    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad is False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]

        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)

        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class GAThead(nn.Module):
    def __init__(self, in_features, dims, activation=F.leaky_relu):
        super(GAThead, self).__init__()
        self.in_features = in_features
        self.dims = dims
        self.linear = nn.Linear(in_features, dims)
        self.linear_att = nn.Linear(dims, 1)
        self.special_spmm = SpecialSpmm()
        self.activation = activation

    def forward(self, x, adj, dropout=0):
        x = F.dropout(x, dropout)
        x = self.linear(x)
        value = self.linear_att(x)
        value = F.leaky_relu(value)
        value = 20 - F.leaky_relu(20 - value)
        value = torch.exp(value)

        divide_factor = torch.spmm(adj, value)
        x = x * value
        x = torch.spmm(adj, x)
        x = x / divide_factor
        if self.activation is not None:
            x = self.activation(x)

        return x


class GATConv(nn.Module):
    def __init__(self, in_features, n_heads, out_features, activation=None, mode=0):
        super(GATConv, self).__init__()
        self.in_features = in_features
        self.n_heads = n_heads
        self.out_features = out_features
        self.heads = nn.ModuleList()
        self.mode = mode
        for i in range(n_heads):
            self.heads.append(GAThead(in_features, out_features, activation=activation))

    def forward(self, x, adj, dropout=0):
        xp = []
        for i in range(self.n_heads):
            xp.append(self.heads[i](x, adj, dropout))
        if self.mode == 0:
            s = torch.cat(xp, 1)
        else:
            s = torch.sum(torch.stack(xp), 0) / self.n_heads

        return s


class GAT(nn.Module):
    def __init__(self, num_layers, num_heads, head_dims, activation=F.leaky_relu):
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dims = head_dims
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            if i != num_layers - 1:
                self.layers.append(
                    GATConv(num_heads[i] * head_dims[i], num_heads[i + 1], head_dims[i + 1], activation=activation))
            else:
                self.layers.append(GATConv(num_heads[i] * head_dims[i], num_heads[i + 1], head_dims[i + 1], mode=1))

    def forward(self, x, adj, dropout=0):
        for layer in self.layers:
            x = layer(x, adj, dropout=dropout)

        return x
