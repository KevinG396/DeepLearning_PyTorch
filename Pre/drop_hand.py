# Use Noised Data  == Tikhonov Normalization
# Dropout: add noise between layers
# Dropout: x' = x + noise, we want E[x'] = x. ==> xi'=0(with p) or xi/(1-p)

import torch
from torch import nn
import cpu_data_n_op as tb

def dropout_layer(X,dropout):
    assert 0<=dropout<=1
    if dropout==1:
        return torch.zero_like(X)
    if dropout==0:
        return X
    mask = (torch.randn(X.shape)>dropout).float()  # random vector, if element>dropout, set to 1, else, set to 0
    return mask*X/(1.0-dropout)

num_inputs, num_outputs, num_hidden1, num_hidden2 = 784,10,256,256
dropout1, dropout2 = 0.2, 0.5

class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hidden1, num_hidden2, 
                 is_training = True):
        super(Net, self).__init__()
        self.num_inputs = num_inputs
        self.training = is_training
        self.lin1 = nn.Linear(num_inputs,num_hidden1)   # define shape
        self.lin2 = nn.Linear(num_hidden1,num_hidden2)
        self.lin3 = nn.Linear(num_hidden2,num_outputs)
        self.relu = nn.ReLU()
    def forward(self,X):
        H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))
        if self.training==True:
            H1 = dropout_layer(H1, dropout1)
        H2 = self.relu(self.lin2(H1))
        if self.training==True:
            H2 = dropout_layer(H2, dropout2)
        out = self.lin3(H2)
        return out

net = Net(num_inputs, num_outputs, num_hidden1, num_hidden2)

if __name__=='__main__':
    num_epochs, lr, batch_size = 10,0.5,256
    loss = nn.CrossEntropyLoss()
    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    train_iter, test_iter = tb.load_data_fashion_mnist(batch_size)
    tb.train(net, train_iter,test_iter,loss,num_epochs,trainer)
