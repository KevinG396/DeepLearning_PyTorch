import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l

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


def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1,keepdim=True)
    return X_exp/partition #broadcast

# a regression model
def net(W,b,X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])),W)+b) #-1 means let program to self-calculate


#y = torch.tensor([0,2])   # catagory number of target vector
#y_hat = torch.tensor([[0.1,0.3,0.6],[0.3,0.2,0.5]])  # target vector tensor   
#print(y_hat[[0,1],y])     # for targets 0 and 1, we get the right catagory


def cross_entropy(y_hat,y):
    return -torch.log(y_hat[range(len(y_hat)),y])

def accuracy(y_hat,y):
    if len(y_hat.shape)>1 and y_hat.shape[1]>1:   #y+hat's dimension >= 2, and the classes number >1
        y_hat = y_hat.argmax(axis=1)   # get the bigget in every row (we will get a vector of number of targets)
    cmp = y_hat.type(y.dtype) == y     # convert y_hat's datatype to the same as y, and compare
    return float(cmp.type(y.dtype).sum())  # total correct pred

#cmp = y_hat.argmax(axis=1)==y
#print(cmp.type(y.dtype).sum())

class Accumulator:
    def __init__(self,n):
        self.data = [0.0]*n
    def add(self, *args): #*arg means we dont need to know how many params in advance
        self.data = [a + float(b) for a,b in zip(self.data, args)]
    def reset(self):
        self.data = [0.0] *len(self.data)
    def __getitem__(self,idx):
        return self.data[idx] 

#metri = Accumulator(2)
#print(metri.__getitem__(1))

def evaluate_accuracy(W,b,net,data_iter):
    if isinstance(net, torch.nn.Module): # if our model is build bu torch.nn.Module
        net.eval()   # change net to eval mode (just forward, no gradient calculation)
    metric = Accumulator(2) 
    for X, y in data_iter:
        metric.add(accuracy(net(W,b,X),y), y.numel()) # calculate right pred and total(y.numel())
    return metric[0] / metric[1]   # right pred / total

def train_epoch_ch3(W,b,lr, net, train_iter, loss, updater):
    if isinstance(net, torch.nn.Module): 
        net.train()    # start training mode
    metric = Accumulator(3) 
    for X, y in train_iter:
        y_hat = net(W,b,X)
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
            updater(W,b,lr,X.shape[0])
            metric.add(float(l.sum()), accuracy(y_hat,y), y.numel())
    print("Avg loss: "+str(metric[0]/metric[2]) + "    " 
          +"    ACC: "+str(metric[1]/metric[2]))
    return metric[0]/metric[2], metric[1]/metric[2]  # Sum_loss / sample: avg_loss; accurate num/sample: accuracy

def train_ch3(W,b,lr,net, train_iter, test_iter, loss, num_epochs,updater):
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(W,b,lr,net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(W,b,net, test_iter)
    train_loss, train_iter = train_metrics
    print("test_acc: "+str(test_acc))


def updater(W,b,lr,batch_size):
    return d2l.sgd([W,b], lr,batch_size)

if __name__=='__main__':

    lr1 = 0.1
    num_epochs1 = 60    # train 60 cycles
    batch_size1 = 8192
    num_inputs1 = 784   #28*28
    num_outputs1 = 10
    W1 = torch.normal(0,0.01,size=(num_inputs1, num_outputs1), requires_grad = True)
    b1 = torch.zeros(num_outputs1, requires_grad = True)
    train_iter1, test_iter1 = load_data_fashion_mnist(batch_size1)
    train_ch3(W1, b1, lr1, net, train_iter1, test_iter1, cross_entropy, num_epochs1, updater)
