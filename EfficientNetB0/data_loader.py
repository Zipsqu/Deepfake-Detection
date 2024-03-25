import os
import json
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = self.load_samples()

    def load_samples(self):
        samples = []
        for filename in os.listdir(self.root_dir):
            if filename.endswith('_face.jpg'):
                image_path = os.path.join(self.root_dir, filename)
                json_path = os.path.join(self.root_dir, filename.replace('_face.jpg', '_metadata.json'))
                if os.path.exists(json_path):
                    samples.append((image_path, json_path))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, json_path = self.samples[idx]
        image = Image.open(image_path).convert('RGB')
        with open(json_path) as f:
            data = json.load(f)
        # Convert label to binary (assuming 'label' is present in JSON)
        label = 1 if data['label'] == 'fake' else 0

        if self.transform:
            image = self.transform(image)
        return image, label


# Define Albumentations augmentations
def get_augmentations():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


if __name__ == "__main__":
    train_root_dir = 'D:/DFDC Sample Dataset/Pre-processed Dataset/Training'
    train_dataset = CustomDataset(train_root_dir, transform=get_augmentations())

    val_root_dir = 'D:/DFDC Sample Dataset/Pre-processed Dataset/Validation'
    val_dataset = CustomDataset(val_root_dir, transform=get_augmentations())

    # Example usage
    for image, label in train_dataset:
        print(image.shape, label)

    for image, label in val_dataset:
        print(image.shape, label)
