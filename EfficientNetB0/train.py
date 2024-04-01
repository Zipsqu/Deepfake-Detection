import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet
from data_loader import CustomDataset, get_augmentations

# Define DataLoader and model parameters
train_root_dir = 'D:/Dataset/Training'
val_root_dir = 'D:/Dataset/Validation'
num_classes = 2  # Binary classification (real or fake)
batch_size = 32
num_epochs = 10
learning_rate = 0.001
weight_decay = 1e-5  # Weight decay coefficient
dropout_rate = 0.5  # Dropout probability
patience = 3  # Early stopping patience

# Create custom datasets and data loaders for training and validation
train_dataset = CustomDataset(train_root_dir, transform=get_augmentations())
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = CustomDataset(val_root_dir, transform=get_augmentations())
val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Define the EfficientNet model with dropout
model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=num_classes)
model._dropout = nn.Dropout(p=dropout_rate)  

# Define loss function and optimizer with weight decay
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

best_val_loss = float('inf')  # Initialize best validation loss
no_improvement_count = 0  # Initialize count for early stopping

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_data_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        # Print batch progress
        print_freq = 10
        if (batch_idx + 1) % print_freq == 0:
            accuracy = 100 * correct / total
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_data_loader)}], Loss: {running_loss / print_freq:.4f}, Accuracy: {accuracy:.2f}%")
            running_loss = 0.0  
            correct = 0 
            total = 0 

    # Check for early stopping
    if val_epoch_loss < best_val_loss:
        best_val_loss = val_epoch_loss
        no_improvement_count = 0
        torch.save(model.state_dict(), 'D:/Dataset/Weights.pth')  # Save the model
    else:
        no_improvement_count += 1
        if no_improvement_count >= patience:
            print(f"Validation loss hasn't improved for {patience} epochs. Early stopping...")
            break

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
torch.save(model.state_dict(), 'D:/DFDC Sample Dataset/Weights.pth')
