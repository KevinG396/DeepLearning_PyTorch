import torch
from torch import nn
import data_n_op as tb

def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides,padding),
        nn.ReLU(), nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU(), nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU()
    )


net = nn.Sequential(
        nin_block(1, 96, kernel_size=11, strides=4, padding=0),
        nn.MaxPool2d(3, stride=2),
        nin_block(96, 256, kernel_size=5, strides=1, padding=2),
        nn.MaxPool2d(3, stride=2),
        nin_block(256, 384, kernel_size=3, strides=1, padding=1),
        nn.MaxPool2d(3, stride=2), nn.Dropout(0.5),
        nin_block(384, 10, kernel_size=3, strides=1, padding=1),
        nn.AdaptiveAvgPool2d((1,1)),
        nn.Flatten()   # 4d tensor with last two dim==1, flatten to 2d
    )

# Check your model
# X = torch.rand(size = (1,1,224,224), dtype=torch.float32)
# for layer in net:
#     X = layer(X)
#     print(layer.__class__.__name__, 'output shape: \t', X.shape)

if __name__ == '__main__':
    lr, num_epochs = 0.1, 10
    batch_size = 128
    train_iter, test_iter = tb.load_data_fashion_mnist(batch_size, resize=224)
    tb.train_gpu(net, train_iter, test_iter, num_epochs, lr, tb.try_gpu())
