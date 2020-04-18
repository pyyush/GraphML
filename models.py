import torch
from layers import GraphConv


class GCN(torch.nn.Module):
    def __init__(self, fan_in, nhid, nclass, dropout=0.2, normalize=True):
        super(GCN, self).__init__()
        self.fan_in = fan_in
        self.nhid = nhid
        self.nclass = nclass
        self.normalize = normalize
        self.drop = torch.nn.Dropout(p=dropout)
        self.lnorm = torch.nn.LayerNorm(normalized_shape=self.nhid)

        # layers
        self.gcn1 = GraphConv(self.fan_in, self.nhid)
        self.gcn2 = GraphConv(self.nhid, self.nhid)
        self.gcn3 = GraphConv(self.nhid, self.nhid)
        self.gcn4 = GraphConv(self.nhid, self.nclass)
    
    def forward(self, sparse_adj, feats):
        
        feats = self.gcn1(sparse_adj, feats)
        if self.normalize:
            feats = self.lnorm(feats)
        feats = torch.nn.functional.relu(feats)
        #feats = self.drop(feats)

        #####################################
        
        feats = self.gcn2(sparse_adj, feats)
        
        if self.normalize:
            feats = self.lnorm(feats)
        feats = torch.nn.functional.relu(feats)
        #feats = self.drop(feats)

        #####################################
        
        feats = self.gcn3(sparse_adj, feats)
        
        if self.normalize:
            feats = self.lnorm(feats)
        feats = torch.nn.functional.relu(feats)
        #feats = self.drop(feats)

        #####################################
        
        out = self.gcn4(sparse_adj, feats)
        
        return torch.nn.functional.log_softmax(out, dim=1)
    
