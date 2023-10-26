import numpy as np
import os
from PIL import Image
from torch.utils.data.dataset import Dataset
from matplotlib.pyplot import imread
import torch
from torch import Tensor
import torch.utils.data as data_utils
import torchvision.transforms.functional as TF
from torchvision import transforms, datasets


def val_test_split(dataset, val_size=5000, batch_size=512, num_workers=2, pin_memory=False):
    # Split into val and test sets
    test_size = len(dataset) - val_size
    dataset_val, dataset_test = data_utils.random_split(
        dataset, (val_size, test_size), generator=torch.Generator().manual_seed(42)
    )
    val_loader = data_utils.DataLoader(dataset_val, batch_size=batch_size, shuffle=False,
                                       num_workers=num_workers, pin_memory=pin_memory)
    test_loader = data_utils.DataLoader(dataset_test, batch_size=batch_size, shuffle=False,
                                        num_workers=num_workers, pin_memory=pin_memory)
    return val_loader, test_loader


def get_mnist_loaders(data_path, batch_size=512, model_class='LeNet',
                      train_batch_size=128, val_size=2000, download=False, device='cpu'):
    if model_class == "MLP":
        tforms = transforms.Compose([transforms.ToTensor(), ReshapeTransform((-1,))])
    else:
        tforms = transforms.ToTensor()

    train_set = datasets.MNIST(data_path, train=True, transform=tforms,
                               download=download)
    val_test_set = datasets.MNIST(data_path, train=False, transform=tforms,
                                  download=download)

    Xys = [train_set[i] for i in range(len(train_set))]
    Xs = torch.stack([e[0] for e in Xys]).to(device)
    ys = torch.stack([torch.tensor(e[1]) for e in Xys]).to(device)
    train_loader = FastTensorDataLoader(Xs, ys, batch_size=batch_size, shuffle=True)
    val_loader, test_loader = val_test_split(val_test_set,
                                             batch_size=batch_size,
                                             val_size=val_size)

    return train_loader, val_loader, test_loader


def get_rotated_mnist_loaders(angle, data_path, model_class='LeNet', download=False):
    if model_class == "MLP":
        shift_tforms = transforms.Compose([RotationTransform(angle), transforms.ToTensor(),
                                           ReshapeTransform((-1,))])
    else:
        shift_tforms = transforms.Compose([RotationTransform(angle), transforms.ToTensor()])

    # Get rotated MNIST val/test sets and loaders
    rotated_mnist_val_test_set = datasets.MNIST(data_path, train=False,
                                                transform=shift_tforms,
                                                download=download)
    shift_val_loader, shift_test_loader = val_test_split(rotated_mnist_val_test_set,
                                                         val_size=2000)

    return shift_val_loader, shift_test_loader


class ReshapeTransform:
    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, img):
        return torch.reshape(img, self.new_size)


class RotationTransform:
    """Rotate the given angle."""

    def __init__(self, angle):
        self.angle = angle

    def __call__(self, x):
        return TF.rotate(x, self.angle)


def uniform_noise(dataset, delta=1, size=5000, batch_size=512):
    if dataset in ['MNIST', 'FMNIST', 'R-MNIST']:
        shape = (1, 28, 28)
    elif dataset in ['SVHN', 'CIFAR10', 'CIFAR100', 'CIFAR-10-C']:
        shape = (3, 32, 32)
    elif dataset in ['ImageNet', 'ImageNet-C']:
        shape = (3, 256, 256)

    # data = torch.rand((100*batch_size,) + shape)
    data = delta * torch.rand((size,) + shape)
    train = data_utils.TensorDataset(data, torch.zeros_like(data))
    loader = torch.utils.data.DataLoader(train, batch_size=batch_size,
                                         shuffle=False, num_workers=1)
    return loader


class FastTensorDataLoader:
    """
    Source: https://github.com/hcarlens/pytorch-tabular/blob/master/fast_tensor_data_loader.py
    and https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """

    def __init__(self, *tensors, batch_size=32, shuffle=False):
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors
        self.dataset = tensors[0]

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i:self.i + self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches


def get_mnist_ood_loaders(ood_dataset, data_path='./data', batch_size=512, download=False):
    '''Get out-of-distribution val/test sets and val/test loaders (in-distribution: MNIST/FMNIST)'''
    tforms = transforms.ToTensor()
    if ood_dataset == 'FMNIST':
        fmnist_val_test_set = datasets.FashionMNIST(data_path, train=False,
                                                    transform=tforms,
                                                    download=download)
        val_loader, test_loader = val_test_split(fmnist_val_test_set,
                                                 batch_size=batch_size,
                                                 val_size=0)
    elif ood_dataset == 'EMNIST':
        emnist_val_test_set = datasets.EMNIST(data_path, split='digits', train=False,
                                              transform=tforms,
                                              download=download)
        val_loader, test_loader = val_test_split(emnist_val_test_set,
                                                 batch_size=batch_size,
                                                 val_size=0)
    elif ood_dataset == 'KMNIST':
        kmnist_val_test_set = datasets.KMNIST(data_path, train=False,
                                              transform=tforms,
                                              download=download)
        val_loader, test_loader = val_test_split(kmnist_val_test_set,
                                                 batch_size=batch_size,
                                                 val_size=0)
    elif ood_dataset == 'MNIST':
        mnist_val_test_set = datasets.MNIST(data_path, train=False,
                                            transform=tforms,
                                            download=download)
        val_loader, test_loader = val_test_split(mnist_val_test_set,
                                                 batch_size=batch_size,
                                                 val_size=0)
    else:
        raise ValueError('Choose one out of FMNIST, EMNIST, MNIST, and KMNIST.')
    return val_loader, test_loader


# TODO needs root data folder and a torch dataloader wrapper
# Creating a sub class of torch.utils.data.dataset.Dataset
class notMNIST(Dataset):

    # The init method is called when this class will be instantiated.
    def __init__(self, root):
        Images, Y = [], []
        folders = os.listdir(root)

        for folder in folders:
            folder_path = os.path.join(root, folder)
            for ims in os.listdir(folder_path):
                try:
                    img_path = os.path.join(folder_path, ims)
                    Images.append(np.array(imread(img_path)))
                    Y.append(ord(folder) - 65)  # Folders are A-J so labels will be 0-9
                except:
                    # Some images in the dataset are damaged
                    print("File {}/{} is broken".format(folder, ims))
        data = [(x, y) for x, y in zip(Images, Y)]
        self.data = data

    # The number of items in the dataset
    def __len__(self):
        return len(self.data)

    # The Dataloader is a generator that repeatedly calls the getitem method.
    # getitem is supposed to return (X, Y) for the specified index.
    def __getitem__(self, index):
        img = self.data[index][0]

        # 8 bit images. Scale between [0,1]. This helps speed up our training
        img = img.reshape(28, 28) / 255.0

        # Input for Conv2D should be Channels x Height x Width
        img_tensor = Tensor(img).view(1, 28, 28).float()
        label = self.data[index][1]
        return (img_tensor, label)


def get_cifar10_loaders(data_path, batch_size=512, val_size=2000,
                        train_batch_size=128, download=False, data_augmentation=True):
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]

    tforms = [transforms.ToTensor(),
              transforms.Normalize(mean, std)]
    tforms_test = transforms.Compose(tforms)
    if data_augmentation:
        tforms_train = transforms.Compose([transforms.RandomHorizontalFlip(),
                                           transforms.RandomCrop(32, padding=4)]
                                          + tforms)
    else:
        tforms_train = tforms_test

    # Get datasets and data loaders
    train_set = datasets.CIFAR10(data_path, train=True, transform=tforms_train,
                                 download=download)
    # train_set = data_utils.Subset(train_set, range(500))
    val_test_set = datasets.CIFAR10(data_path, train=False, transform=tforms_test,
                                    download=download)

    train_loader = data_utils.DataLoader(train_set,
                                         batch_size=train_batch_size,
                                         shuffle=True,num_workers=2)
    val_loader, test_loader = val_test_split(val_test_set,
                                             batch_size=batch_size,
                                             val_size=val_size)

    return train_loader, val_loader, test_loader

def get_cifar100_loaders(data_path, batch_size=512, val_size=2000,
                        train_batch_size=128, download=False, data_augmentation=True):
    mean = [x / 255 for x in [129.31, 124.1, 112.40]]
    std = [x / 255 for x in [68.21, 65.40, 70.40]]

    tforms = [transforms.ToTensor(),
              transforms.Normalize(mean, std)]
    tforms_test = transforms.Compose(tforms)
    if data_augmentation:
        tforms_train = transforms.Compose([transforms.RandomHorizontalFlip(),
                                           transforms.RandomCrop(32, padding=4)]
                                          + tforms)
    else:
        tforms_train = tforms_test

    # Get datasets and data loaders
    train_set = datasets.CIFAR100(data_path, train=True, transform=tforms_train,
                                 download=download)
    # train_set = data_utils.Subset(train_set, range(500))
    val_test_set = datasets.CIFAR100(data_path, train=False, transform=tforms_test,
                                    download=download)

    train_loader = data_utils.DataLoader(train_set,
                                         batch_size=train_batch_size,
                                         shuffle=True,num_workers=2)
    val_loader, test_loader = val_test_split(val_test_set,
                                             batch_size=batch_size,
                                             val_size=val_size)

    return train_loader, val_loader, test_loader

class DatafeedImage(torch.utils.data.Dataset):
    def __init__(self, x_train, y_train, transform=None):
        self.x_train = x_train
        self.y_train = y_train
        self.transform = transform

    def __getitem__(self, index):
        img = self.x_train[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, self.y_train[index]

    def __len__(self):
        return len(self.x_train)


def load_corrupted_cifar10(severity, data_dir='data', batch_size=256, cuda=True,
                           workers=2):
    """ load corrupted CIFAR10 dataset """

    x_file = data_dir + '/CIFAR-10-C/CIFAR10_c%d.npy' % severity
    np_x = np.load(x_file)
    y_file = data_dir + '/CIFAR-10-C/CIFAR10_c_labels.npy'
    np_y = np.load(y_file).astype(np.int64)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    dataset = DatafeedImage(np_x, np_y, transform)
    dataset = data_utils.Subset(dataset, torch.randint(len(dataset), (10000,)))

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=cuda)

    return loader


def get_cifar10_ood_loaders(ood_dataset, data_path='./data', batch_size=512, download=False):
    '''Get out-of-distribution val/test sets and val/test loaders (in-distribution: CIFAR-10)'''
    if ood_dataset == 'SVHN':
        svhn_tforms = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.4376821, 0.4437697, 0.47280442),
                                                   (0.19803012, 0.20101562, 0.19703614))])
        svhn_val_test_set = datasets.SVHN(data_path, split='test',
                                          transform=svhn_tforms,
                                          download=download)
        val_loader, test_loader = val_test_split(svhn_val_test_set,
                                                 batch_size=batch_size,
                                                 val_size=0)
    elif ood_dataset == 'LSUN':
        lsun_tforms = transforms.Compose([transforms.Resize(size=(32, 32)),
                                          transforms.ToTensor()])
        lsun_test_set = datasets.LSUN(data_path, classes=['classroom_val'],  # classes='test'
                                      transform=lsun_tforms)
        val_loader = None
        test_loader = data_utils.DataLoader(lsun_test_set, batch_size=batch_size,
                                            shuffle=False)
    elif ood_dataset == 'CIFAR-100':
        cifar100_tforms = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                   (0.2023, 0.1994, 0.2010))])
        cifar100_val_test_set = datasets.CIFAR100(data_path, train=False,
                                                  transform=cifar100_tforms,
                                                  download=download)
        val_loader, test_loader = val_test_split(cifar100_val_test_set,
                                                 batch_size=batch_size,
                                                 val_size=0)
    else:
        raise ValueError('Choose one out of SVHN, LSUN, and CIFAR-100.')
    return val_loader, test_loader
