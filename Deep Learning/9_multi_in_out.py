import torch
from torch import nn
import data_n_op as tb

def cov2d(X,K):
    # X is the original image, K is the kernel
    h,w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w +1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i,j] = (X[i:i+h, j:j+w] * K).sum()
    return Y

def corr2d_multi_in(X,K):  # assume X and K are both 3d (3 channel img, 3 channel kernel)
    return sum(cov2d(x,k) for x,k in zip(X,K)) # sum all channels to one

def corr2d_multi_in_out(X,K):  # K is already 4d (c_out, c_in, h, w)
    return torch.stack([corr2d_multi_in(X,k) for k in K],0) # stack all channels


X = torch.tensor([[[0.0,1.0,2.0],[3.0,4.0,5.0],[6.0,7.0,8.0]],    # img_channel_1
                   [[1.0,2.0,3.0],[4.0,5.0,6.0],[7.0,8.0,9.0]]])  # img_channel_2
K = torch.tensor([[[0.0,1.0],[2.0,3.0]],    # kernel_channel_1
                  [[1.0,2.0],[3.0,4.0]]])   # kernel_channel_2

print("Multi channel img's output: \n",corr2d_multi_in(X,K), '\n')

K = torch.stack((K,K+1,K+2), 0)  # 3 output channel
print("Kernel shape (c_out, c_in, h, w): ", K.shape, '\n')

print("Multi_channel out: \n", corr2d_multi_in_out(X, K), '\n')


