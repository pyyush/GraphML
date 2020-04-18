import torch
import math
import torch.nn.init as init
from torch.nn.parameter import Parameter


class GraphConv(torch.nn.Module):
    
    """ GCN layer, X' = (A'X)W + b """

    def __init__(self, in_features, out_features):
        super(GraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # init weights & biases
        self.weight = Parameter(torch.Tensor(self.out_features, self.in_features))
        self.bias = Parameter(torch.Tensor(self.out_features))
        self.reset_parameters()
    

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)


    def forward(self, A, X):

        S = torch.sparse.mm(A, X)
        output = S.matmul(self.weight.t())
        
        if self.bias is not None:
            output += self.bias
            
        return output
        
        
        
        
