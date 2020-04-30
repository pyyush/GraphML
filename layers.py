import math
import torch
import torch.nn.init as init
from torch.nn.parameter import Parameter


class GraphConv(torch.nn.Module):
    """ Performes Graph Convolution operation on incoming data X'^(l + 1) = (A'X^(l))W^(l) + b
    
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
          additional dimensions and :math:`H_{in} = \text{in\_features}`
        - Output: :math:`(N, *, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
                :math:`(\text{out\_features}, \text{in\_features})`. The values are
                initialized from :math:`\mathcal{U}(-\text{bound}, \text{bound})` where
                :math: `\text{bound} = \text{gain} \times \sqrt{\frac{3}{\text{fan\_mode}}}`
        bias: the learnable bias of the module of shape
              :math:`(\text{out\_features})`.
              If :attr:`bias` is ``True``, the values are initialized with the scalar value `0`.
    
    """
    
    __constants__ = ['in_features, out_features']

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(self.out_features, self.in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(self.out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, nonlinearity='relu') 
        if self.bias is not None:
            init.zeros_(self.bias)


    def forward(self, A, X):

        S = torch.sparse.mm(A, X)
        output = S.matmul(self.weight.t())
        
        if self.bias is not None:
            output += self.bias
            
        return output
        
        
    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(self.in_features, self.out_features, self.bias is not None)
        
