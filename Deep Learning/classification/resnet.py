import torch
from torch import nn
from torch.nn import functional as F
import data_n_op as tb

class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, 
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, 
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, 
                               kernel_size=3, padding=1)

        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)  # 'inplace' reduce memory use
    
    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X   # Merge input features to new features so that new features can include old features
        return F.relu(Y)

'''
# residual blocks that do not change size
blk = Residual(3,3, use_1x1conv=True)
X = torch.rand(4, 3, 6, 6)
Y = blk(X)
print(Y.shape)

# residual blocks that change size
blk = Residual(3,6, use_1x1conv=True, strides=2)
blk(X).shape
'''

b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))


def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i==0 and not first_block:
            blk.append(
                Residual(input_channels, num_channels, use_1x1conv=True, strides=2)
            )
        else:
            blk.append(Residual(num_channels, num_channels))
            # do not need to set conv1x1 to true because dim_input==dim_output
    return blk

# '*' is used to depart resnet_block to several residuals(nn.Modules) 
b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))

net = nn.Sequential(b1, b2, b3, b4, b5,
                    nn.AdaptiveAvgPool2d((1,1)),
                    nn.Flatten(), nn.Linear(512, 10))

'''
X = torch.rand(size=(1, 1, 224, 224))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)
'''

if __name__ == '__main__':
    lr, num_epochs = 0.06, 20
    batch_size = 600
    train_iter, test_iter = tb.load_data_fashion_mnist(batch_size, resize=96)
    tb.train_gpu(net, train_iter, test_iter, num_epochs, lr, tb.try_gpu())
