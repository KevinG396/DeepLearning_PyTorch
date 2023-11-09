import torch
from torch import nn
import data_n_op as tb


# cov2d actrually correlation2d
def cov2d(X,K):
    # X is the original image, K is the kernel
    h,w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w +1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i,j] = (X[i:i+h, j:j+w] * K).sum()
    return Y

# conv2d layer
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return cov2d(x, self.weight) + self.bias

# example: edge detection
X = torch.ones((6,8))
X[:,2:6] = 0
K = torch.tensor([[1.0,-1.0]])
Y = cov2d(X,K)
print("Edge detection example: \n","X: \n",X,'\n',"Y: \n",Y)


# Given Images and Edge Images, how to train a kernel?
conv2d = nn.Conv2d(1,1,kernel_size=(1,2), bias = False) # input channel = output channel = 1
X = X.reshape((1,1,6,8))
Y = Y.reshape((1,1,6,7))
print("\nTrain a kernel: \n")
for i in range(10):
    Y_h = conv2d(X)
    loss = (Y_h - Y)**2  #MSE
    conv2d.zero_grad()
    loss.sum().backward()
    conv2d.weight.data[:] -= 3e-2 * conv2d.weight.grad
    print(f'batch {i+1}, loss {loss.sum():.3f}')
print(conv2d.weight.data.reshape(1,2),'\n')

def comp_conv2d(conv2d, X):
    X = X.reshape((1,1) + X.shape) # add batch size and number of channels
    Y = conv2d(X)
    return Y.reshape(Y.shape[2:]) # check the single img output of conv layer

conv2d_1 = nn.Conv2d(1,1, kernel_size = 3, padding = 1)
X_1 = torch.rand(size = (8,8))
print("8*8 through padding_conv output: ",comp_conv2d(conv2d_1,X_1).shape)

conv2d_2 = nn.Conv2d(1,1,kernel_size = 3, padding = 1, stride = 2)
print("8*8 through padding_stride_conv output: ",comp_conv2d(conv2d_2,X_1).shape)

