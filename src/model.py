import torch
import torch.nn as nn
from .module import MaskedLinear


class MLP(nn.module):
    def __init__(self, n_in, n_out, n_hiddens, depth=1, activator='tanh'):
        super().__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.n_hiddens = n_hiddens
        self.depth = depth
        if activator == 'tanh':
            self.activator = torch.tanh()
        elif activator == 'relu':
            self.activator = torch.relu()
        
        self.construction()

    def construction(self):
        net = [nn.Linear(self.n_in, self.n_hiddens), self.activator]
        for i in range(self.depth-1):
            net.extend([
                nn.Linear(self.n_hiddens, self.n_hiddens),
                self.activator
                ])
        net.extend([
            nn.Linear(self.n_hiddens, self.n_out),
            ])

        self.net = nn.Sequential(*[net])
    
    def forward(self, x):
        return self.net(x)


class MADE(nn.Module):
    def __init__(self, n, depth, width, activator='tanh', residual=False):
        '''
            n: 自旋个数，为网络的输入输出神经元个数
            depth: 网络深度
        '''
        super(MADE, self).__init__()
        self.n = n
        self.depth = depth
        self.n_hiddens = depth-1
        self.width = width
        if activator=='tanh':
            self.activator = nn.Tanh()
        elif activator=='relu':
            self.activator = nn.ReLU()
        elif activator=='sigmoid':
            self.activator = nn.Sigmoid()
        self.residual=residual
        self.construction(width, depth)


    def construction(self, width, depth):
        n = self.n
        net = []
        net.extend([
            MaskedLinear(n, 1, 1 if depth==0 and width==1 else width, False), 
            self.activator,
            ])
        for i in range(depth):
            net.extend([
                MaskedLinear(n, width, width, True, True),
                self.activator,
                ])
        if width != 1:
            net.extend([
                MaskedLinear(n, width, 1, True, True),
                self.activator,
                ])
        net.pop()
        
        net.extend([nn.Sigmoid(),])
        self.net = nn.Sequential(*net)
        

    def forward(self, x):
        return self.net(x)
    
    def log_prob(self, samples):
        a = 1e-30
        s = self.forward(samples)
        mask = (samples + 1)/2
        mask = mask.view(mask.shape[0], - 1)
        log_p = (torch.log(s+a) * mask + torch.log(1 - s+a) * (1 - mask)).sum(dim=1)
        return log_p
  

    def partial_forward(self, n_s, condition, device, dtype, k=1):
        with torch.no_grad():
            if n_s >1 :
                m = condition.size(1)
            else:
                m = condition.size(0)
            x = torch.zeros(n_s, self.n, device=device, dtype=dtype)
            x[:, :m] = condition
            for i in range(int(2*k)):
                s_hat = self.forward(x)
                x[:, m+i] = torch.floor(2*s_hat[:, m+i]) * 2 - 1
        return x
    

    def partial_samples(self, n_s, condition, device, dtype):
        with torch.no_grad():
            m = condition.size(0)
            x = torch.zeros(n_s, self.n, device=device, dtype=dtype)
            x[:, :m] = torch.vstack([condition]*n_s)
            for i in range(self.n-m):
                s_hat = self.forward(x)
                x[:, m+i] = torch.bernoulli(s_hat[:, m+i]) * 2 - 1
        return x
    


    