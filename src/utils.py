import torch
import torch.nn as nn
import torch.nn.functional as F

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
    