import torch
from torch import nn
import data_n_op as tb

class Reshape(torch.nn.Module):
    def forward(self, x):
        return x.view(-1,1,28,28)  # unchanged_batch_size, 1 channel, 28*28 img 
    
net = torch.nn.Sequential(
    Reshape(), 
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16*5*5, 120), nn.Sigmoid(),
    nn.Linear(120 ,84), nn.Sigmoid(),
    nn.Linear(84,10)
)

# Check your model
# X = torch.rand(size = (1,1,28,28), dtype=torch.float32)
# for layer in net:
#     X = layer(X)
#     print(layer.__class__.__name__, 'output shape: \t', X.shape)


if __name__ == '__main__':
    lr, num_epochs = 0.9, 10
    batch_size = 256
    train_iter, test_iter = tb.load_data_fashion_mnist(batch_size)
    tb.train_gpu(net, train_iter, test_iter, num_epochs, lr, tb.try_gpu())