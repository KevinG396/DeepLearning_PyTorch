import torch
from torch import nn
import cpu_data_n_op as tb


net = nn.Sequential(nn.Flatten(), nn.Linear(784,10)) #Flatten any dims to 2D: btach and vector

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)
    
net.apply(init_weights)

loss = nn.CrossEntropyLoss()

trainer = torch.optim.SGD(net.parameters(), lr=0.1)

if __name__=='__main__':
    num_epochs = 10
    batch_size = 256
    train_iter, test_iter = tb.load_data_fashion_mnist(batch_size)
    tb.train(net, train_iter,test_iter,loss,num_epochs,trainer)
