import os
import torch
from torchvision import transforms
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10, ImageFolder

from dataloader.datasets import get_cifar_anomaly_dataset
from dataloader.datasets import get_mnist_anomaly_dataset
# from dataloader.kdd_dataset import get_loader
from dataloader.kdd import KDD_dataset
import numpy as np

class Data:
    """ Dataloader containing train and valid sets.
    """
    def __init__(self, train, valid):
        self.train = train
        self.valid = valid

##
def load_data(opt):
    """ Load Data

    Args:
        opt ([type]): Argument Parser

    Raises:
        IOError: Cannot Load Dataset

    Returns:
        [type]: dataloader
    """

    ##
    # LOAD DATA SET
    if opt.dataroot == '':
        opt.dataroot = './data/{}'.format(opt.dataset)

    ## CIFAR
    if opt.dataset in ['cifar10']:
        transform = transforms.Compose([transforms.Resize(opt.isize),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_ds = CIFAR10(root='./data', train=True, download=True, transform=transform)
        valid_ds = CIFAR10(root='./data', train=False, download=True, transform=transform)
        train_ds, valid_ds = get_cifar_anomaly_dataset(train_ds, valid_ds, int(opt.abnormal_class))

    ## MNIST
    elif opt.dataset in ['mnist']:
        transform = transforms.Compose([transforms.Resize(opt.isize),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))])

        train_ds = MNIST(root='./data', train=True, download=True, transform=transform)
        valid_ds = MNIST(root='./data', train=False, download=True, transform=transform)
        train_ds, valid_ds = get_mnist_anomaly_dataset(train_ds, valid_ds, int(opt.abnormal_class))

    # FOLDER
    elif opt.dataset in ['OCT']:
        # TODO: fix the OCT dataset into the dataloader and return
        class OverLapCrop():
            def __init__(self, img_size):
                self.img_size = img_size
            def __call__(self, x):
                ret = []
                for i in range(256//opt.isize):
                    for j in range(256//opt.isize):
                        ret.append(x[:, i*opt.isize:(i+1)*opt.isize, j*opt.isize:(j+1)*opt.isize])
                return ret
        splits = ['train', 'test']
        drop_last_batch = {'train': True, 'test': True}
        shuffle = {'train': True, 'test': False}
        train_transform = transforms.Compose([transforms.Grayscale(),
                                              transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                                            transforms.Resize([256,256]),
                                            transforms.RandomCrop(opt.isize), #
                                            transforms.ToTensor(),])

        test_transform = transforms.Compose([transforms.Grayscale(),
                                        transforms.Resize([256,256]),
                                        transforms.ToTensor(),
                                        OverLapCrop(opt.isize),
                                        transforms.Lambda(lambda crops: torch.stack(crops)),])

        dataset = {'train': ImageFolder(os.path.join(opt.dataroot, 'train'), train_transform),
                   'test':ImageFolder(os.path.join(opt.dataroot, 'test'), test_transform),}

        train_dl = DataLoader(dataset=dataset['train'],
                             batch_size=opt.batchsize,
                             shuffle=shuffle['train'],
                             num_workers=int(opt.n_cpu),
                             drop_last=drop_last_batch['train'],
                             worker_init_fn=(None if 42 == -1
                             else lambda x: np.random.seed(42)))
        valid_dl = DataLoader(dataset=dataset['test'],
                             batch_size=opt.batchsize,
                             shuffle=shuffle['test'],
                             num_workers=int(opt.n_cpu),
                             drop_last=drop_last_batch['test'],
                             worker_init_fn=(None if 42 == -1
                             else lambda x: np.random.seed(42)))
        return Data(train_dl, valid_dl)

    elif opt.dataset in ['KDD99']:
        train_ds = KDD_dataset(opt, mode='train')
        valid_ds = KDD_dataset(opt, mode='test')

    else:
        raise NotImplementedError

    ## DATALOADER
    train_dl = DataLoader(dataset=train_ds, batch_size=opt.batchsize, shuffle=True, drop_last=True)
    valid_dl = DataLoader(dataset=valid_ds, batch_size=opt.batchsize, shuffle=False, drop_last=False)

    return Data(train_dl, valid_dl)

