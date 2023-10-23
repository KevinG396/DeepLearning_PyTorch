import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l
from matplotlib import pyplot as plt

#d2l.use_svg_display() #display imgs

trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(root="Fasion_data", 
                                               train = True,
                                               transform=trans, # imgs to Tensor
                                               download = True)

mnist_test = torchvision.datasets.FashionMNIST(root="Fasion_data", train = False,transform=trans,download = True)

#print(len(mnist_test), len(mnist_train))
#print(mnist_train[0][0].shape) # [0][0] ==> example[0]'s img (example = img(tensor) + target(int))

def get_fashion_mnist_labels(labels):
    text_labels = [
        't-shirt','trouser','pullover','dress','coat','sandal','shirt','sneaker','bag','ankle boot'
    ]
    return [text_labels[int(i)] for i in labels]

def show_images(imgs,num_rows, num_cols, titles=None, scale=1.5):
    figsize = (num_cols*scale, num_rows*scale)
    _,axes = d2l.plt.subplots(num_rows,num_cols,figsize=figsize)
    axes = axes.flatten()
    for i,(ax,img) in enumerate(zip(axes,imgs)):
        if torch.is_tensor(img):
            ax.imshow(img.numpy())
            ax.axis('off')
            ax.set_title(titles[i])
        else:
            ax.imshow(img)

#X,y = next(iter(data.DataLoader(mnist_train, batch_size=18)))

#show_images(X.reshape(18,28,28),2,9,titles=get_fashion_mnist_labels(y))
#plt.show()

batch_size = 256

if __name__ == '__main__':
    train_iter = data.DataLoader(mnist_train, batch_size, shuffle = True, num_workers = 4)
    timer = d2l.Timer()
    for X,y in train_iter:
        continue
    print(f'{timer.stop():.2f} sec')
