# initialize w = 0 and b = 0
# repeat
#   if y_i[<w, x_i> + b] <= 0 then:
#       w <- w + y_i*x_i and b <- b + y_i
#   end if
# until all classified correctly 
#
# Loss(y,x,w) = max(0, -y<w,x>) if T, -y<w,x> < 0; else, -y<w,x> > 0

import torch
from torch import nn
import data_n_op as tb

batch_size = 256
train_iter, test_iter = tb.load_data_fashion_mnist(batch_size)

num_inputs, num_outputs, num_hiddens = 784,10,256

W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens, requires_grad=True)) # parameters generating randomly
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs, requires_grad=True))
b2 = nn.Parameter(torch.randn(num_outputs, requires_grad=True))

params = [W1,b1,W2,b2]

def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X,a)

def net(X):
    X = X.reshape((-1,num_inputs))
    H = relu(X@W1+b1)
    return (H@W2+b2)

loss = nn.CrossEntropyLoss()
trainer = torch.optim.SGD(params, lr=0.1)
num_epochs = 10

if __name__ == '__main__':
    tb.train(net, train_iter,test_iter,loss,num_epochs,trainer)
