import torch
from torch import nn
import data_n_op as tb


# moving mean and moving var are pmean and var on the dataset, not batch
# momentum is used to update moving mean and moving var, fixed
def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    
    # if we are inferencing
    if not torch.is_grad_enabled():
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    
    # if we are training
    else:
        assert len(X.shape) in (2,4)   # 2 means fc layer; 4 means conv layer
        if len(X.shape)==2:
            # dim0: batch size; dim1: feature
            mean = X.mean(dim=0)  # affact on feature dim
            var = ((X - mean)**2).mean(dim=0)
        else:
            # dim0: batch size; dim1: channels; dim2,3: h, w
            mean = X.mean(dim=(0,2,3), keepdim=True)  # affact on channel dim
            var = ((X - mean)**2).mean(dim=(0,2,3), keepdim=True)
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # make moving_mean converge to real moving_mean(real mean of dataset)
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta
    return Y, moving_mean.data, moving_var.data

class BatchNorm(nn.Module):
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims==2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # variables in nn.Parameters will be updated with the optimizer 
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)
    
    def forward(self, X):
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.beta, self.moving_mean, self.moving_var, eps=1e-5, momentum=0.9)
        return Y

net = torch.nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5),
    BatchNorm(6, num_dims=4),    # or nn.BatchNorm2d(6)
    nn.Sigmoid(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), 
    BatchNorm(16, num_dims=4), 
    nn.Sigmoid(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16*4*4, 120), 
    BatchNorm(120, num_dims=2), 
    nn.Sigmoid(),
    nn.Linear(120 ,84), 
    BatchNorm(84, num_dims=2), 
    nn.Sigmoid(),
    nn.Linear(84,10)
)


if __name__ == '__main__':
    lr, num_epochs = 0.9, 10
    batch_size = 256
    train_iter, test_iter = tb.load_data_fashion_mnist(batch_size)
    tb.train_gpu(net, train_iter, test_iter, num_epochs, lr, tb.try_gpu())
