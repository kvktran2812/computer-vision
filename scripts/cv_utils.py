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


def model_parameters(model):
    total_params = sum([p.numel() for p in model.parameters()])
    print(f"Total number of parameters: {total_params/1e6:.4f}M")
    return total_params

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

def train(
    model, 
    train_loader, 
    test_loader,
    optimizer, 
    criterion, 
    device, 
    model_name: str,
    n_epochs:int=10, 
    save_checkpoint:int = 1,
    save_dir: str = "models"
):
    print("Training model on device: ", device)

    model.to(device)
    epoch_loss_history = []
    batch_loss_history = []

    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # loss update
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        # validation
        model.eval()
        train_acc = 100. * correct / total

        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for val_images, val_labels in test_loader:
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                val_outputs = model(val_images)
                loss = criterion(val_outputs, val_labels)
                val_loss += loss.item() 
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_acc = 100. * correct / total
        
        train_loss = running_loss / len(train_loader) 
        val_loss /= len(test_loader)
        epoch_loss_history.append(train_loss)
        print(f"Epoch {epoch+1}/{n_epochs}: Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
        print(f"    Train Acc: {train_acc:.2f}% - Val Acc: {val_acc:.2f}%")
        print("#" * 20)

        # save model per checkpoint
        if epoch % save_checkpoint == 0:
            torch.save(model.state_dict(), f"{save_dir}/{model_name}/checkpoint_{(epoch) / save_checkpoint}.pth")
            print(f"Saving model checkpoint to {save_dir}/{model_name}/checkpoint_{(epoch) / save_checkpoint}.pth")

    # save final model
    torch.save(model.state_dict(), f"{save_dir}/{model_name}/checkpoint_final.pth")
    return epoch_loss_history, batch_loss_history