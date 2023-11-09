import torch
from torch import nn
import data_n_op as tb


net = torch.nn.Sequential(
    nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Flatten(),
    nn.Linear(6400, 4096), nn.ReLU(), nn.Dropout(p=0.5),
    nn.Linear(4096 ,4096), nn.ReLU(), nn.Dropout(p=0.5),
    nn.Linear(4096,100)
)

# Check your model
# X = torch.rand(size = (1,1,224,224), dtype=torch.float32)
# for layer in net:
#     X = layer(X)
#     print(layer.__class__.__name__, 'output shape: \t', X.shape)


if __name__ == '__main__':
    lr, num_epochs = 0.06, 30
    batch_size = 1200
    train_iter, test_iter = tb.load_CIFAR100(batch_size, resize=224)
    tb.train_gpu(net, train_iter, test_iter, num_epochs, lr, tb.try_gpu())