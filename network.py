import torch
from torch import nn
from torch.nn import functional as F
import cpu_data_n_op as tb

# new net define
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(784, 256)
        self.out = nn.Linear(256,10)
    
    def forward(self, X):
        X = X.reshape((-1,784))
        return self.out(F.relu(self.hidden(X)))

# new layer define is same as new net define
# new layer (or new net) can be put into nn.Sequential()

dropout1, dropout2 = 0.2, 0.5

Net = nn.Sequential(
    nn.Flatten(), 
    nn.Linear(784,256), nn.ReLU(), nn.Dropout(dropout1),
    nn.Linear(256,256), nn.ReLU(), nn.Dropout(dropout2),
    nn.Linear(256,10)
)

#net = Net
#print(net[2].state_dict())   # 3rd layer info

# embedded blocks
def block1():
    return nn.Sequential(nn.Linear(4,8), nn.ReLU(), nn.Linear(8,4), nn.ReLU())

def block2():
    net = nn.Sequential()
    for i in range(4):
        net.add_module(f'block {i}', block1())
    return net

def init_weights_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)

def xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

# net[2].apply(xavier)

# Shared params and params binding

shared = nn.Linear(8,8)
net_eg = nn.Sequential(nn.Linear(4,8), nn.ReLU(), shared, 
                       nn.ReLU(), shared, nn.ReLU(), nn.Linear(8,2))
# no metter how params change, net_eg[2] == net_eg[4]

if __name__ == '__main__':
    net = MLP()
    # Load params
    net.load_state_dict(torch.load("mlp_NN.params"))
    print(net.eval())
    num_epochs = 10
    batch_size = 256
    lr = 0.3
    loss = nn.CrossEntropyLoss()
    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    train_iter, test_iter = tb.load_data_fashion_mnist(batch_size)
    tb.train(net, train_iter,test_iter,loss,num_epochs,trainer)
    torch.save(net.state_dict(), 'mlp_NN.params')