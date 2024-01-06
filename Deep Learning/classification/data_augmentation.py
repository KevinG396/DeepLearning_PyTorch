import torch
import torchvision
from torch import nn
import data_n_op as tb
from d2l import torch as d2l

train_augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor()
])

test = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

def load_CIFAR10(is_train, aug_method, batch_size):
    dataset = torchvision.datasets.CIFAR10(root="CIFAR10", 
                                            train = is_train,
                                            transform=aug_method, # data augmentation
                                            download = True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                    shuffle=is_train, num_workers=4)
    return dataloader


batch_size, device, net = 256, tb.try_gpu(),  d2l.resnet18(10, 3)

def init_weights(m):
    if type(m) is [nn.Linear, nn.Conv2d]:
        nn.init.xavier_uniform_(m.weight)

net. apply(init_weights)


def train_with_data_aug(train_augs, test, net, lr=0.001):
    train_iter = load_CIFAR10(True, train_augs, batch_size)
    test_iter = load_CIFAR10(False, test, batch_size)
    loss = nn.CrossEntropyLoss(reduction="none")
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    tb.train_gpu(net, train_iter, test_iter, 10, lr, tb.try_gpu())

if __name__ == '__main__':
    train_with_data_aug(train_augs, test, net)
