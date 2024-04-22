import torch
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet
from data_loader2 import CustomDataset, get_augmentations
from sklearn.metrics import classification_report

# Define evaluation dataset parameters
eval_root_dir = 'D:/Dataset/Evaluation'
num_classes = 2  
batch_size = 32

# Create the evaluation dataset and data loader
eval_dataset = CustomDataset(eval_root_dir, transform=get_augmentations())
eval_data_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

# Define the EfficientNet model
model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=num_classes)

# Load the trained weights
model.load_state_dict(torch.load('D:/Dataset/weights.pt'))

# Set the model to evaluation mode
model.eval()

# Initialize lists to store model predictions and ground truth labels
eval_predictions = []
eval_ground_truth = []

# Device configuration (assuming you're using GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move model to the device
model.to(device)

# Iterate through the evaluation dataset
with torch.no_grad():
    for eval_images, eval_labels in eval_data_loader:
        eval_images, eval_labels = eval_images.to(device), eval_labels.to(device)
        eval_outputs = model(eval_images)
        _, eval_predicted = torch.max(eval_outputs, 1)

        # Append model predictions and ground truth labels to lists
        eval_predictions.extend(eval_predicted.cpu().numpy().tolist())
        eval_ground_truth.extend(eval_labels.cpu().numpy().tolist())

# Compute classification report
report = classification_report(eval_ground_truth, eval_predictions, target_names=['real', 'fake'])
print("Classification Report:")
print(report)
