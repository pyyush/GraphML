import math
import torch
import torch.nn.init as init
from torch.nn.parameter import Parameter


class GraphConv(torch.nn.Module):
    
    """GCN layer, X^(l + 1) = (AX^(l))W^(l) + b """

    def __init__(self, in_features, out_features, bias=True, init='kaiming'):
        super(GraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.init = init
        
        # init weights & biases
        self.weight = Parameter(torch.Tensor(self.out_features, self.in_features))

        if bias:
            self.bias = Parameter(torch.Tensor(self.out_features))
        self.reset_parameters()
    

    def reset_parameters(self):

        if self.init == 'xavier':
            init.xavier_uniform_(self.weight, gain=init.calculate_gain('relu'))

        elif self.init == 'kaiming':
            init.kaiming_uniform_(self.weight, nonlinearity='relu') 

        if self.bias is not None:
            init.zeros_(self.bias)


    def forward(self, A, X):

        S = torch.sparse.mm(A, X)
        output = S.matmul(self.weight.t())
        
        if self.bias is not None:
            output += self.bias
            
        return output
        
        
        
        
