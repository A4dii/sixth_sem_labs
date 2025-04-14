#ALL CODE TOGETHER
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
import os

# -----------------------------
# Section 1: Regularization Experiments
# -----------------------------

# 1. Data Augmentation Experiment:
#    - Use transforms.RandomRotation, RandomHorizontalFlip, etc.
#    - Compare training performance with and without augmentation

def get_data_loaders(batch_size=32, data_aug=False):
    data_dir = './cats_and_dogs_filtered'
    # Define transforms
    if data_aug:
        train_transforms = transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            # Optionally, add noise here manually if desired
        ])
    else:
        train_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    valid_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Load the dataset (assumes directory structure follows ImageFolder format)
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transforms)
    valid_dataset = datasets.ImageFolder(os.path.join(data_dir, 'validation'), transform=valid_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader

# 2. Define a simple CNN for cat-dog classification
class CatDogCNN(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(CatDogCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # conv layer
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 56 * 56, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # Using built-in dropout
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return out

# 3a. L2 Regularization using optimizer’s weight_decay parameter
def train_with_l2_weight_decay(model, train_loader, valid_loader, epochs=10, weight_decay=1e-4):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=weight_decay)

    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        # Add validation and logging here
        print(f"Epoch {epoch+1}/{epochs} completed with weight_decay L2")

# 3b. L2 Regularization using manual loop to add L2 norm
def train_with_l2_manual(model, train_loader, valid_loader, epochs=10, lambda_l2=1e-4):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            # Manual L2 regularization: iterate over parameters
            l2_norm = sum(torch.norm(param) ** 2 for param in model.parameters())
            loss += lambda_l2 * l2_norm
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs} completed with manual L2")

# 4. L1 Regularization using manual loop
def train_with_l1_manual(model, train_loader, valid_loader, epochs=10, lambda_l1=1e-4):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            # Manual L1 regularization: iterate over parameters
            l1_norm = sum(torch.sum(torch.abs(param)) for param in model.parameters())
            loss += lambda_l1 * l1_norm
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs} completed with manual L1")

# 5. Dropout Regularization
#    - Built-in dropout is already shown in the model above.
#    - Here, we define a custom dropout layer.
class CustomDropout(nn.Module):
    def __init__(self, p=0.5):
        super(CustomDropout, self).__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        # Create a mask using Bernoulli distribution
        mask = torch.bernoulli(torch.ones_like(x) * (1 - self.p))
        return mask * x / (1 - self.p)

# To compare, you can modify the classifier of your model to use CustomDropout:
class CatDogCNN_CustomDropout(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(CatDogCNN_CustomDropout, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 56 * 56, 128),
            nn.ReLU(),
            CustomDropout(dropout_rate),  # Custom dropout
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return out

# 6. Early Stopping Implementation
def train_with_early_stopping(model, train_loader, valid_loader, epochs=50, patience=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_val_loss = float('inf')
    trigger_times = 0

    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in valid_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        val_loss /= len(valid_loader)
        print(f"Epoch {epoch+1}: Validation Loss: {val_loss:.4f}")

        # Check for early stopping condition
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print("Early stopping triggered")
                break

# 7. Hyperparameter Experiments:
#    - You can wrap the training in loops that iterate through various lambda values or dropout rates.
def hyperparameter_experiment():
    reg_strengths = [1e-3, 1e-4, 1e-5]
    for reg in reg_strengths:
        print(f"Training with L2 lambda: {reg}")
        model = CatDogCNN()
        train_loader, valid_loader = get_data_loaders(batch_size=32, data_aug=False)
        train_with_l2_manual(model, train_loader, valid_loader, epochs=5, lambda_l2=reg)

def train_model(model, train_loader, valid_loader, criterion, optimizer, epochs=10):
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            # Adjust this code for your specific task and input data format
            inputs, targets = batch
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        # Validation loop can be added here
        print(f"Epoch {epoch+1} completed.")

    # Example: Run data augmentation vs. non-augmentation for cat-dog classification
    train_loader_aug, valid_loader_aug = get_data_loaders(batch_size=32, data_aug=True)
    train_loader_no_aug, valid_loader_no_aug = get_data_loaders(batch_size=32, data_aug=False)

    model = CatDogCNN(dropout_rate=0.5)
    print("Training with data augmentation:")
    train_with_l2_weight_decay(model, train_loader_aug, valid_loader_aug, epochs=5, weight_decay=1e-4)

    model = CatDogCNN(dropout_rate=0.5)
    print("Training without data augmentation:")
    train_model(model, train_loader_no_aug, valid_loader_no_aug, nn.CrossEntropyLoss(), optim.Adam(model.parameters(), lr=0.001), epochs=5)

    # Example: Run L2 and L1 regularization experiments
    # Uncomment and run the following experiments as needed:
    # train_with_l2_manual(model, train_loader_no_aug, valid_loader_no_aug, epochs=5, lambda_l2=1e-4)
    # train_with_l1_manual(model, train_loader_no_aug, valid_loader_no_aug, epochs=5, lambda_l1=1e-4)

    # Example: Early stopping
    # train_with_early_stopping(model, train_loader_no_aug, valid_loader_no_aug, epochs=50, patience=3)

    # Example: Hyperparameter experiment on L2 regularization strength
    # hyperparameter_experiment()