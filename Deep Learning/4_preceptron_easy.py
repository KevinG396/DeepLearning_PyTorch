# VC dimension: 2D input preceptron, VC dim=3 (4 pionts xor can not be divided)


import torch
from torch import nn
import data_n_op as tb


num_inputs, num_outputs, num_hiddens_1 = 784,10,256

net = nn.Sequential(nn.Flatten(), nn.Linear(num_inputs, num_hiddens_1), 
                    nn.ReLU(), nn.Linear(num_hiddens_1,num_outputs))

net.apply(tb.init_weights)

loss = nn.CrossEntropyLoss()

if __name__ == '__main__':
    num_epochs = 10
    batch_size = 256
    lr = 0.3
    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    train_iter, test_iter = tb.load_data_fashion_mnist(batch_size)
    tb.train(net, train_iter,test_iter,loss,num_epochs,trainer)
    torch.save(net.state_dict(), 'mlp.params')