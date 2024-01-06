import torch
from torch import nn
import data_n_op as tb

dropout1, dropout2 = 0.2, 0.5

net = nn.Sequential(
    nn.Flatten(), 
    nn.Linear(784,256), nn.ReLU(), nn.Dropout(dropout1),
    nn.Linear(256,256), nn.ReLU(), nn.Dropout(dropout2),
    nn.Linear(256,10)
)

net.apply(tb.init_weights)
loss = nn.CrossEntropyLoss()

if __name__ == '__main__':
    num_epochs = 10
    batch_size = 256
    lr = 0.3
    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    train_iter, test_iter = tb.load_data_fashion_mnist(batch_size)
    tb.train(net, train_iter,test_iter,loss,num_epochs,trainer)
