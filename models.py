import torch
from layers import GraphConv


class GCN(torch.nn.Module):
    def __init__(self, fan_in, fan_out, layers, dropout=0.2, normalize=True, bias=False):
        super(GCN, self).__init__()
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.num_layers = layers
        self.dropout = dropout
        self.normalize = normalize
        self.bias = bias
        self.network = torch.nn.ModuleList([GraphConv(in_features=fan_in[i], out_features=fan_out[i], dropout=self.dropout, bias=self.bias, normalize=self.normalize) for i in range(self.num_layers - 1)])
        self.network.append(GraphConv(in_features=fan_in[-1], out_features=fan_out[-1], bias=self.bias, last=True))
        
                                            
    def forward(self, sparse_adj, feats):
        for idx, layer in enumerate(self.network):
            feats = layer(sparse_adj, feats)
        return feats
