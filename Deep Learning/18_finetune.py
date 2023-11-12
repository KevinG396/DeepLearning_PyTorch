import os
import torch
from torch import nn
import torchvision
from torch.nn import functional as F
import data_n_op as tb

normalize = torchvision.transforms.Normalize(
    [0.485, 0.456, 0.406],    # rgb mean normalize
    [0.229, 0.224, 0.225])    # rgb std normalize

train_augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(), normalize
])

test_augs = torchvision.transforms.Compose([
    torchvision.transforms.Resize([256,256]),  # crop to 256,256
    torchvision.transforms.CenterCrop(224), # crop to 224,224
    torchvision.transforms.ToTensor(), normalize
])

pretrained_net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)

finetune_net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
finetune_net.fc = nn.Linear(finetune_net.fc.in_features, 10)
nn.init.xavier_uniform_(finetune_net.fc.weight)

def load_CIFAR10(is_train, aug_method, batch_size):
    dataset = torchvision.datasets.CIFAR10(root="CIFAR10", 
                                            train = is_train,
                                            transform=aug_method, # data augmentation
                                            download = True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                    shuffle=is_train, num_workers=4)
    return dataloader

def fine_tune_train(net, lr, batch_size=256, num_epoch=10, param_group=True):
    train_iter = load_CIFAR10(True, train_augs, batch_size)
    test_iter = load_CIFAR10(False, test_augs, batch_size)
    loss = nn.CrossEntropyLoss(reduction="none")
    if param_group:
        # params_1x mean all layers except last layer (fc)
        params_1x = [param for name, param in net.named_parameters()
             if name not in ["fc.weight", "fc.bias"]]
        '''
        params_1x_name = [name for name, param in net.named_parameters()]
        print(params_1x_name)
        '''
        # other layers use low lr, last fc layer use larger lr
        trainer = torch.optim.SGD([{'params': params_1x},
                                   {'params': net.fc.parameters(),
                                    'lr': lr * 10}],
                                lr=lr, weight_decay=0.001)
    else:
        trainer = torch.optim.SGD(net.parameters(), lr=lr,
                                  weight_decay=0.001)
    tb.train_gpu(net, train_iter, test_iter, 10, lr, tb.try_gpu())


if __name__ == '__main__':
    # for layer in finetune_net.children():
    #     print(layer)
    fine_tune_train(finetune_net, 5e-5)
