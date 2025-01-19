import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

def cifar_transform():
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
    ])
    return transform

def download_cifar10(transform, root='./cifar-10', download=True, batch_size=64, num_workers=2):
    train_dataset = torchvision.datasets.CIFAR10(root=root, train=True, download=download, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root=root, train=False, download=download, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader

def download_cifar100(transform, root='./cifar-100', download=True, batch_size=64, num_workers=2):
    train_dataset = torchvision.datasets.CIFAR100(root=root, train=True, download=download, transform=transform)
    test_dataset = torchvision.datasets.CIFAR100(root=root, train=False, download=download, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader

