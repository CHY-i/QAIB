import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MaskedLinear(nn.Linear):
    """ same as Linear except has a configurable mask on the weights """
    
    def __init__(self, n, in_features, out_features, self_connection=True, bias=True):
        '''n: 自旋个数,
           n*in: 总的输入个数,
           n*out: 总的输出个数,
         '''
        super(MaskedLinear, self).__init__(n * in_features, n * out_features, bias)
        #定义一个名为mask个的buffer     
        if self_connection:
            self.register_buffer('mask', torch.tril(torch.ones(n, n)))#注意 pytorch中是用行向量乘W.T定义的线性运算
        else:
            self.register_buffer('mask', torch.tril(torch.ones(n, n), diagonal=-1))
        self.mask = torch.cat([self.mask] * in_features, dim=1)
        self.mask = torch.cat([self.mask] * out_features, dim=0)
        self.weight.data *= self.mask
        if n !=1 :
            self.weight.data *= torch.sqrt(self.mask.numel() / self.mask.sum())
    def forward(self, input):
            return F.linear(input, self.weight*self.mask, self.bias)

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # [batch, seq, dim] 
        seq_len = x.size(1)
        positions = self.pe[:seq_len]  
        return x + positions.unsqueeze(0)

