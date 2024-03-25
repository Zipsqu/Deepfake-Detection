import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet
from data_loader import CustomDataset, get_augmentations

# Define DataLoader and model parameters
train_root_dir = 'D:/DFDC Sample Dataset/Pre-processed Dataset/Training'
val_root_dir = 'D:/DFDC Sample Dataset/Pre-processed Dataset/Validation'
num_classes = 2  # Binary classification (real or fake)
batch_size = 32
num_epochs = 10
learning_rate = 0.001

# Create custom datasets and data loaders for training and validation
train_dataset = CustomDataset(train_root_dir, transform=get_augmentations())
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = CustomDataset(val_root_dir, transform=get_augmentations())
val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Define the EfficientNet model
model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=num_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_data_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        # Print batch progress
        if (i + 1) % 10 == 0:  # Print every 10 batches
            print(
                f'Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}/{len(train_data_loader)}], Loss: {loss.item():.4f}')

    epoch_loss = running_loss / len(train_dataset)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {epoch_loss:.4f}")

    # Validation loop
    model.eval()
    val_running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():
        for val_images, val_labels in val_data_loader:
            val_images, val_labels = val_images.to(device), val_labels.to(device)
            val_outputs = model(val_images)
            val_loss = criterion(val_outputs, val_labels)
            val_running_loss += val_loss.item() * val_images.size(0)

            _, predicted = torch.max(val_outputs, 1)
            correct_predictions += (predicted == val_labels).sum().item()
            total_predictions += val_labels.size(0)

    val_epoch_loss = val_running_loss / len(val_dataset)
    val_accuracy = correct_predictions / total_predictions
    print(
        f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

# Save trained model weights
torch.save(model.state_dict(), 'D:/DFDC Sample Dataset/Weights')
