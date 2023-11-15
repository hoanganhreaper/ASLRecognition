import torch
from dataset.CustomDataset import ASLCustomDataset, transform
from torch.utils.data import DataLoader
from torch import nn 
from modelalexnet import ModelAlexNet
from torch import optim

# Create Data Loader
batch_size = 32

train_dataset = ASLCustomDataset(root_dir='asl_alphabet_train\\asl_alphabet_train', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

val_dataset = ASLCustomDataset(root_dir='asl_alphabet_train\\asl_alphabet_train', transform=transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

num_classes = 29
model = ModelAlexNet(num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Validation loop
model.eval()
with torch.no_grad():
    for inputs, labels in val_loader:
        outputs = model(inputs)
        # Evaluate model performance