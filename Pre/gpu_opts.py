import torch
from torch import nn

def try_gpu(i=0):
    if torch.cuda.device_count() >= i+1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

x = torch.ones(2,3,device = try_gpu())
y = torch.rand(2,3,device = try_gpu(2))

z = x + y.cuda(0) # y is on cpu, COPY y to gpu
print(z)

net = nn.Sequential(nn.Linear(3,1))
net = net.to(device=try_gpu())

print(net(x))
