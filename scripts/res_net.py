import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# variables 
train_dir = "../tiny-imagenet-200/train"
val_dir = "../tiny-imagenet-200/val"
test_dir = "../tiny-imagenet-200/test"
save_dir = "../models/res_net"

# hyperparameters
n_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# data preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),      
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = ImageFolder(root=train_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

print(f"Dataset size: {len(train_dataset)}")


# ResNet implementation
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.identity_mapping = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.identity_mapping = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = self.identity_mapping(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + identity
        x = self.relu(x)
        return x
    
class ResNet34(nn.Module):
    def __init__(self, num_classes=200):
        super(ResNet34, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1 = self.make_layer(in_channels=64, out_channels=64, num_blocks=3, stride=1)
        self.layer2 = self.make_layer(in_channels=64, out_channels=128, num_blocks=4, stride=2)
        self.layer3 = self.make_layer(in_channels=128, out_channels=256, num_blocks=6, stride=2)
        self.layer4 = self.make_layer(in_channels=256, out_channels=512, num_blocks=3, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(Block(in_channels, out_channels, stride))  # Downsampling block
        for _ in range(1, num_blocks):
            layers.append(Block(out_channels, out_channels))  # Identity blocks
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
    


# model initialization
model = ResNet34(num_classes=200).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

for epoch in range(n_epochs):
    model.train()
    running_loss = 0.0
    
    for idx, (image, label) in enumerate(train_loader):
        image, label = image.to(device), label.to(device)
        optimizer.zero_grad()
        outputs = model(image)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
    
        running_loss += loss.item()
    
        if idx % 100 == 0:
            print(f"Epoch {epoch+1}/{n_epochs} - Step {idx}: Loss {running_loss/idx:4f}")


    # loss and evaluation
    train_loss = running_loss / len(train_loader) 
    print(f"Final Eval - Epoch {epoch+1}/{n_epochs} - Train Loss: {train_loss:.4f}")
    print("#" * 20)

    scheduler.step()

    # save model
    if epoch % 10 == 0:
        torch.save(model.state_dict(), f"{save_dir}{epoch}.pth")