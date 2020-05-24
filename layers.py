import math
import torch
from torch.nn.parameter import Parameter


class GraphConv(torch.nn.Module):
    """ Applies the Graph Convolution operation to the incoming data: math: `X' = \hat{A}XW`
    
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        dropout: dropout probability
            Default: 0.2
        bias: If set to ``True ``, the layer will learn an additive bias.
            Default: ``False``
        normalize: If set to ``False``, the layer will not apply Layer Normalization to the features.
            Default: ``True``
        last: If set to ``True``, the layer will act as the final/classification layer and apply log_softmax non-linear activation function.
            Default: ``False`` 
    Shape:
        - Input: :math:`(N, H_{in})` where :math:`H_{in} = \text{in\_features}`
        - Output: :math:`(N, H_{out})` where :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
                :math:`(\text{out\_features}, \text{in\_features})`. The values are
                initialized from :math:`\mathcal{U}(-\text{bound}, \text{bound})` where
                :math: `\text{bound} = \sqrt{\frac{6}{\text{in\_features + out\_features}}}`
        bias: the learnable bias of the module of shape
              :math:`(\text{out\_features})`.
              If :attr:`bias` is ``True``, the values are initialized with the scalar value `0`.
    
    """
    
    __constants__ = ['in_features, out_features']

    def __init__(self, in_features, out_features, dropout=0.2, bias=False, normalize=True, last=False):
        super(GraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.normalize = normalize
        self.p = dropout
        self.last = last
        self.weight = Parameter(torch.Tensor(self.out_features, self.in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(self.out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
   
    def reset_parameters(self):
        # Xavier Glorot Uniform
        #bound = math.sqrt(6.0/float(self.out_features + self.in_features))
        #self.weight.data.uniform_(-bound, bound)
        
        # Kaiming He Uniform
        torch.nn.init.kaiming_uniform_(self.weight, a=0.01, mode='fan_in', nonlinearity='leaky_relu')
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, A, X):

        input = torch.sparse.mm(A, X) # (N, N) x (N, F) -> (N, F)
        #output = input.matmul(self.weight.t()) # (N, F) x (W, N).t() -> (N, H)
        #if self.bias is not None:
            #output += self.bias
        output = torch.nn.functional.linear(input, self.weight, self.bias) 
        
        if self.last:
            return torch.nn.functional.log_softmax(output, dim=1)
        
        if self.normalize:
            output = torch.layer_norm(output, normalized_shape=self.out_features)
        output = torch.nn.functional.leaky_relu(output)
        return torch.nn.functional.dropout(output, p=self.p, training=self.training)
        
    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(self.in_features, self.out_features, self.bias is not None)
