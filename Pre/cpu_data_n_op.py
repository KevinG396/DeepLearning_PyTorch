# This File is for Fasion_MNIST data loading and train/test functions

import torch
import torchvision
from torch import nn
from torch.utils import data
from torchvision import transforms


def load_data_fashion_mnist(batch_size, resize = None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0,transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root="Fasion_data", 
                                               train = True,
                                               transform=trans, # imgs to Tensor
                                               download = True)
    mnist_test = torchvision.datasets.FashionMNIST(root="Fasion_data", train = False,transform=trans,download = True)

    return(data.DataLoader(mnist_train, batch_size, shuffle = True, num_workers = 4),
           data.DataLoader(mnist_test, batch_size, shuffle = False, num_workers = 4))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

def accuracy(y_hat,y):
    if len(y_hat.shape)>1 and y_hat.shape[1]>1:   #y+hat's dimension >= 2, and the classes number >1
        y_hat = y_hat.argmax(axis=1)   # get the bigget in every row (we will get a vector of number of targets)
    cmp = y_hat.type(y.dtype) == y     # convert y_hat's datatype to the same as y, and compare
    return float(cmp.type(y.dtype).sum())  # total correct pred


class Accumulator:
    def __init__(self,n):
        self.data = [0.0]*n
    def add(self, *args): #*arg means we dont need to know how many params in advance
        self.data = [a + float(b) for a,b in zip(self.data, args)]
    def reset(self):
        self.data = [0.0] *len(self.data)
    def __getitem__(self,idx):
        return self.data[idx] 


def evaluate_accuracy(net,data_iter):
    if isinstance(net, torch.nn.Module): # if our model is build bu torch.nn.Module
        net.eval()   # change net to eval mode (just forward, no gradient calculation)
    metric = Accumulator(2) 
    for X, y in data_iter:
        metric.add(accuracy(net(X),y), y.numel()) # calculate right pred and total(y.numel())
    return metric[0] / metric[1]   # right pred / total


def train_epoch(net, train_iter, loss, updater):
    if isinstance(net, torch.nn.Module): 
        net.train()    # start training mode
    metric = Accumulator(3) 
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer): # if our updater is torch defined optimizor
            updater.zero_grad()
            l.backward()
            updater.step() # update params 
            metric.add(
                float(l) * len(y), accuracy(y_hat, y),
                y.size().numel()
            )
        else:
            l.sum().backward()
            updater(X.shape[0])
            metric.add(float(l.sum()), accuracy(y_hat,y), y.numel())
    print("Avg loss: "+str(metric[0]/metric[2]) + "    " 
          +"    ACC: "+str(metric[1]/metric[2]))
    return metric[0]/metric[2], metric[1]/metric[2]  # Sum_loss / sample: avg_loss; accurate num/sample: accuracy


def train(net, train_iter, test_iter, loss, num_epochs,updater):
    for epoch in range(num_epochs):
        train_metrics = train_epoch(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
    train_loss, train_iter = train_metrics
    print("test_acc: "+str(test_acc))
