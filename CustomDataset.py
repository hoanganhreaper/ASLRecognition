import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class ASLCustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.images = self.load_images()

    def load_images(self):
        images = []
        for cls in self.classes:
            class_path = os.path.join(self.root_dir, cls)
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                images.append((img_path, self.class_to_idx[cls]))
        return images
    
    def __len__ (self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path, label = self.images[idx]

        if os.path.exists(img_path):
            image = Image.open(img_path).convert("RGB")
        else:
            print(f"Image not found: {img_path}")

        if self.transform:
            image = self.transform(image)

        return image, label


# Define Data Transformations
transform = transforms.Compose([
    transforms.Resize((227, 227)),  # AlexNet expects 227x227 input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])





